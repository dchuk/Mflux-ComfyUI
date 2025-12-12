import sys
import torch
import os

def apply_mps_patch():
    # Only run on Mac/MPS
    if not torch.backends.mps.is_available():
        return

    print("🍎 [MFlux] Detecting MacOS/MPS... Scanning for SAM3 modules to patch.")

    decoder_mod = None
    geo_mod = None

    # Scan loaded modules for the specific SAM3 files we know exist
    # based on the directory structure: nodes/sam3_lib/model/...
    for mod in list(sys.modules.values()):
        if not hasattr(mod, '__file__') or not mod.__file__:
            continue

        # Normalize path separators
        fpath = mod.__file__.replace("\\", "/")

        if fpath.endswith("nodes/sam3_lib/model/decoder.py"):
            decoder_mod = mod
        elif fpath.endswith("nodes/sam3_lib/model/geometry_encoders.py"):
            geo_mod = mod

        if decoder_mod and geo_mod:
            break

    if not decoder_mod or not geo_mod:
        print("⚠️ [MFlux] SAM3 modules not found in memory. Patch skipped.")
        return

    print(f"🍎 [MFlux] Found SAM3 Decoder at: {decoder_mod.__file__}")

    # --- DEFINE FIXES ---

    def fix_decoder(mod):
        # Grab dependencies directly from the found module
        box_cxcywh_to_xyxy = mod.box_cxcywh_to_xyxy
        activation_ckpt_wrapper = mod.activation_ckpt_wrapper

        def _get_rpb_matrix_fixed(self, reference_boxes, feat_size):
            H, W = feat_size
            boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes).transpose(0, 1)
            bs, num_queries, _ = boxes_xyxy.shape

            if self.compilable_cord_cache is None:
                self.compilable_cord_cache = self._get_coords(H, W, reference_boxes.device)
                self.compilable_stored_size = (H, W)

            if torch.compiler.is_dynamo_compiling() or self.compilable_stored_size == (H, W):
                coords_h, coords_w = self.compilable_cord_cache
            else:
                if feat_size not in self.coord_cache:
                    self.coord_cache[feat_size] = self._get_coords(H, W, reference_boxes.device)
                coords_h, coords_w = self.coord_cache[feat_size]

            # --- FIX 1: Ensure devices match ---
            if coords_h.device != reference_boxes.device:
                coords_h = coords_h.to(reference_boxes.device)
                coords_w = coords_w.to(reference_boxes.device)
            # -----------------------------------

            deltas_y = coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]
            deltas_y = deltas_y.view(bs, num_queries, -1, 2)
            deltas_x = coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]
            deltas_x = deltas_x.view(bs, num_queries, -1, 2)

            if self.boxRPB in ["log", "both"]:
                deltas_x_log = deltas_x * 8
                deltas_x_log = (torch.sign(deltas_x_log) * torch.log2(torch.abs(deltas_x_log) + 1.0) / np.log2(8))
                deltas_y_log = deltas_y * 8
                deltas_y_log = (torch.sign(deltas_y_log) * torch.log2(torch.abs(deltas_y_log) + 1.0) / np.log2(8))
                if self.boxRPB == "log":
                    deltas_x = deltas_x_log
                    deltas_y = deltas_y_log
                else:
                    deltas_x = torch.cat([deltas_x, deltas_x_log], dim=-1)
                    deltas_y = torch.cat([deltas_y, deltas_y_log], dim=-1)

            act_enable = self.training and self.use_act_checkpoint
            deltas_x = activation_ckpt_wrapper(self.boxRPB_embed_x)(x=deltas_x, act_ckpt_enable=act_enable)
            deltas_y = activation_ckpt_wrapper(self.boxRPB_embed_y)(x=deltas_y, act_ckpt_enable=act_enable)

            B = deltas_y.unsqueeze(3) + deltas_x.unsqueeze(2)
            B = B.flatten(2, 3).permute(0, 3, 1, 2).contiguous()
            return B
        return _get_rpb_matrix_fixed

    def fix_geometry(mod):
        # FIX 2: Removed assert_async
        def concat_padded_sequences_fixed(seq1, mask1, seq2, mask2, return_index: bool = False):
            seq1_length, batch_size, hidden_size = seq1.shape
            seq2_length, batch_size, hidden_size = seq2.shape

            actual_seq1_lengths = (~mask1).sum(dim=-1)
            actual_seq2_lengths = (~mask2).sum(dim=-1)
            final_lengths = actual_seq1_lengths + actual_seq2_lengths
            max_length = seq1_length + seq2_length
            concatenated_mask = (torch.arange(max_length, device=seq2.device)[None].repeat(batch_size, 1) >= final_lengths[:, None])

            concatenated_sequence = torch.zeros((max_length, batch_size, hidden_size), device=seq2.device, dtype=seq2.dtype)
            concatenated_sequence[:seq1_length, :, :] = seq1

            index = torch.arange(seq2_length, device=seq2.device)[:, None].repeat(1, batch_size)
            index = index + actual_seq1_lengths[None]
            concatenated_sequence = concatenated_sequence.scatter(0, index[:, :, None].expand(-1, -1, hidden_size), seq2)

            if return_index:
                return concatenated_sequence, concatenated_mask, index
            return concatenated_sequence, concatenated_mask

        # FIX 3: Empty points check (Fixes your specific crash)
        def _encode_points_fixed(self, points, points_mask, points_labels, img_feats):
            n_points, bs = points.shape[:2]

            # --- CRITICAL FIX: Handle empty points ---
            if n_points == 0:
                return torch.zeros(0, bs, self.d_model, device=points.device), points_mask
            # -----------------------------------------

            points_embed = None
            if self.points_direct_project is not None:
                points_embed = self.points_direct_project(points)

            if self.points_pool_project is not None:
                grid = points.transpose(0, 1).unsqueeze(2)
                grid = (grid * 2) - 1
                sampled = torch.nn.functional.grid_sample(img_feats, grid, align_corners=False)
                sampled = sampled.squeeze(-1).permute(2, 0, 1)
                proj = self.points_pool_project(sampled)
                points_embed = proj if points_embed is None else points_embed + proj

            if self.points_pos_enc_project is not None:
                x, y = points.unbind(-1)
                enc_x, enc_y = self.pos_enc._encode_xy(x.flatten(), y.flatten())
                enc_x = enc_x.view(n_points, bs, enc_x.shape[-1])
                enc_y = enc_y.view(n_points, bs, enc_y.shape[-1])
                proj = self.points_pos_enc_project(torch.cat([enc_x, enc_y], -1))
                points_embed = proj if points_embed is None else points_embed + proj

            type_embed = self.label_embed(points_labels.long())
            return type_embed + points_embed, points_mask

        return concat_padded_sequences_fixed, _encode_points_fixed

    # --- APPLY ---
    try:
        # Patch Decoder
        decoder_mod.TransformerDecoder._get_rpb_matrix = fix_decoder(decoder_mod)

        # Patch Geometry
        concat_fn, encode_fn = fix_geometry(geo_mod)
        geo_mod.concat_padded_sequences = concat_fn
        geo_mod.SequenceGeometryEncoder._encode_points = encode_fn

        print("✅ [MFlux] SAM3 patched successfully for MPS.")
    except Exception as e:
        print(f"❌ [MFlux] Error applying patches: {e}")
        import traceback
        traceback.print_exc()

apply_mps_patch()