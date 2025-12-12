import sys
import torch
import importlib
import os

# Only run on Mac (MPS)
if torch.backends.mps.is_available():
    print("🍎 [MFlux] Detecting MacOS... Attempting to patch SAM3 for compatibility.")

    # 1. Helper to find the SAM3 module regardless of folder name
    def find_sam3_module():
        # Check if already loaded in memory
        for key in sys.modules.keys():
            if "sam3_lib.model.decoder" in key:
                return key.split(".nodes")[0]

        # If not loaded, look in custom_nodes
        try:
            import folder_paths
            base_path = folder_paths.get_folder_paths("custom_nodes")[0]
            for item in os.listdir(base_path):
                if os.path.isdir(os.path.join(base_path, item)):
                    if os.path.exists(os.path.join(base_path, item, "nodes", "sam3_lib", "__init__.py")):
                        return f"custom_nodes.{item}"
        except:
            pass
        return None

    # 2. Define the fixed methods
    def get_fixed_decoder_method(decoder_module):
        box_cxcywh_to_xyxy = decoder_module.box_cxcywh_to_xyxy
        activation_ckpt_wrapper = decoder_module.activation_ckpt_wrapper

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

            # --- FIX: Ensure devices match ---
            if coords_h.device != reference_boxes.device:
                coords_h = coords_h.to(reference_boxes.device)
                coords_w = coords_w.to(reference_boxes.device)
            # ---------------------------------

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

    def get_fixed_geometry_functions(geo_module):
        # FIX: concat_padded_sequences without async assert
        def concat_padded_sequences_fixed(seq1, mask1, seq2, mask2, return_index: bool = False):
            seq1_length, batch_size, hidden_size = seq1.shape
            seq2_length, batch_size, hidden_size = seq2.shape

            # --- FIX: Removed torch._assert_async calls ---

            actual_seq1_lengths = (~mask1).sum(dim=-1)
            actual_seq2_lengths = (~mask2).sum(dim=-1)
            final_lengths = actual_seq1_lengths + actual_seq2_lengths
            max_length = seq1_length + seq2_length

            concatenated_sequence = torch.zeros((max_length, batch_size, hidden_size), device=seq2.device, dtype=seq2.dtype)
            concatenated_sequence[:seq1_length, :, :] = seq1

            index = torch.arange(seq2_length, device=seq2.device)[:, None].repeat(1, batch_size)
            index = index + actual_seq1_lengths[None]
            concatenated_sequence = concatenated_sequence.scatter(0, index[:, :, None].expand(-1, -1, hidden_size), seq2)

            concatenated_mask = (torch.arange(max_length, device=seq2.device)[None].repeat(batch_size, 1) >= final_lengths[:, None])

            if return_index:
                return concatenated_sequence, concatenated_mask, index
            return concatenated_sequence, concatenated_mask

        # FIX: _encode_points with 0-length check
        def _encode_points_fixed(self, points, points_mask, points_labels, img_feats):
            n_points, bs = points.shape[:2]
            # --- FIX: Early return for empty points ---
            if n_points == 0:
                return torch.zeros(0, bs, self.d_model, device=points.device), points_mask

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

    # 3. Apply Patch
    sam3_root = find_sam3_module()
    if sam3_root:
        try:
            import numpy as np # Helper needed inside the closure
            decoder_mod = importlib.import_module(f"{sam3_root}.nodes.sam3_lib.model.decoder")
            geo_mod = importlib.import_module(f"{sam3_root}.nodes.sam3_lib.model.geometry_encoders")

            # Apply patches
            decoder_mod.TransformerDecoder._get_rpb_matrix = get_fixed_decoder_method(decoder_mod)
            concat_fixed, encode_fixed = get_fixed_geometry_functions(geo_mod)
            geo_mod.concat_padded_sequences = concat_fixed
            geo_mod.SequenceGeometryEncoder._encode_points = encode_fixed

            print("✅ [MFlux] SAM3 patched successfully for MPS.")
        except Exception as e:
            print(f"⚠️ [MFlux] Failed to patch SAM3: {e}")
    else:
        print("⚠️ [MFlux] SAM3 nodes not found. Skipping patch.")