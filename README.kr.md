Here is the updated Korean `README.md` file, translated from the current English version provided.

```markdown
<h1 align="center">Mflux-ComfyUI 2.1.0</h1>

<p align="center">
    <strong>mflux 0.13.1 (Apple Silicon/MLX)용 ComfyUI 노드</strong><br/>
    <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

## 개요 (Overview)

이 포크는 ComfyUI 워크플로우 호환성을 유지하면서 기존 노드를 **mflux 0.13.1**을 사용하도록 업그레이드했습니다. mflux 0.13.x의 새로운 통합 아키텍처를 활용하여 표준 FLUX 생성뿐만 아니라 Fill, Depth, Redux, Z-Image Turbo와 같은 특수 변형도 지원합니다.

- **백엔드**: mflux 0.13.1 (macOS + Apple Silicon 필요).
- **그래프 호환성**: 기존 입력이 내부적으로 마이그레이션되므로 이전 그래프가 여전히 작동합니다.
- **통합 로딩**: 로컬 경로, HuggingFace 리포지토리 ID, 사전 정의된 별칭(예: `dev`, `schnell`)을 원활하게 처리합니다.

## mflux 0.13.1의 새로운 기능
이 버전은 중요한 백엔드 개선 사항을 제공합니다:
- **Z-Image Turbo 지원**: 속도에 최적화된 빠른 증류(distilled) Z-Image 변형(6B 파라미터)을 지원합니다.
- **FIBO 및 Qwen 지원**: FIBO 및 Qwen-Image 아키텍처에 대한 백엔드 지원.
- **스마트 모델 로더**: 캐시된 모델과 로컬 모델을 시각적으로 표시하고 폴더를 재귀적으로 스캔합니다.
- **통합 아키텍처**: 모델, LoRA 및 토크나이저에 대한 향상된 해상도(resolution) 지원.

## 주요 기능

- **핵심 생성**: 하나의 노드(`QuickMfluxNode`)에서 빠른 text2img 및 img2img 수행.
- **Z-Image Turbo**: 새로운 고속 모델을 위한 전용 노드 (`MFlux Z-Image Turbo`).
- **하드웨어 최적화**: 메모리가 낮은 Mac에서의 충돌 방지를 위한 **Low RAM** 모드 및 **VAE Tiling** 전용 노드.
- **FLUX 도구 지원**: **Fill** (인페인팅), **Depth** (구조 가이드), **Redux** (이미지 변형)를 위한 전용 노드.
- **ControlNet**: Canny 미리보기 및 최선의(best‑effort) 컨디셔닝; **Upscaler** ControlNet 지원 포함.
- **LoRA 지원**: 통합 LoRA 파이프라인 (LoRA 적용 시 양자화는 반드시 8이어야 함).
- **양자화**: 메모리 효율성을 위한 다양한 옵션 (None, 3, 4, 5, 6, 8-bit).
- **메타데이터**: mflux CLI 도구와 호환되는 전체 생성 메타데이터(PNG + JSON) 저장.

## 설치 (Installation)

### ComfyUI-Manager 사용 (권장)
- "Mflux-ComfyUI"를 검색하여 설치하세요.

### 수동 설치
1. 커스텀 노드 디렉토리로 이동합니다:
   ```bash
   cd /path/to/ComfyUI/custom_nodes
   ```
2. 리포지토리를 복제합니다:
   ```bash
   git clone https://github.com/joonsoome/Mflux-ComfyUI.git
   ```
3. ComfyUI 가상 환경을 활성화하고 의존성을 설치합니다:
   ```bash
   # 표준 venv 예시
   source /path/to/ComfyUI/venv/bin/activate

   pip install --upgrade pip wheel setuptools
   pip install 'mlx>=0.27.0' 'huggingface_hub>=0.26.0'
   pip install 'mflux==0.13.1'
   ```
4. ComfyUI를 재시작합니다.

**참고**: `mflux 0.13.1`은 `mlx >= 0.27.0`이 필요합니다. 이전 버전을 사용 중이라면 업그레이드하세요.

## 노드 설명 (Nodes)

### MFlux/Air (Standard)
- **QuickMfluxNode**: 표준 FLUX txt2img, img2img, LoRA, ControlNet을 위한 올인원 노드.
- **MFlux Z-Image Turbo**: Z-Image 생성 전용 노드 (최적화된 기본값: 9 steps, guidance 없음).
- **Mflux Optimizations**: **Low RAM** (MemorySaver) 및 **VAE Tiling** 설정을 구성하고 메인 노드에 연결합니다.
- **Mflux Models Loader**: 스마트 모델 선택기. `models/Mflux`를 재귀적으로 스캔하고 시스템 캐시를 확인합니다.
  - 🟢 = 캐시됨 (사용 준비 완료)
  - 📁 = 로컬 (ComfyUI 폴더 내)
  - ☁️ = 별칭 (다운로드가 트리거될 수 있음)
- **Mflux Models Downloader**: HuggingFace에서 로컬 폴더로 양자화된 모델 또는 전체 모델을 직접 다운로드합니다.
- **Mflux Custom Models**: 커스텀 양자화 변형을 구성하고 저장합니다.

### MFlux/Pro (Advanced)
- **Mflux Fill**: 인페인팅 및 아웃페인팅을 위한 FLUX.1-Fill 지원 (마스크 필요).
- **Mflux Depth**: 구조 가이드 생성을 위한 FLUX.1-Depth 지원.
- **Mflux Redux**: 이미지 스타일/구조 혼합을 위한 FLUX.1-Redux 지원.
- **Mflux Upscale**: Flux ControlNet Upscaler를 사용한 이미지 업스케일링.
- **Mflux Img2Img / Loras / ControlNet**: 커스텀 파이프라인 구축을 위한 모듈식 로더.

## 사용 팁 (Usage Tips)

- **Z-Image Turbo**: 전용 노드를 사용하세요. 기본값은 **9 steps** 및 **0 guidance**입니다 (이 모델에 필수).
- **최적화**: 메모리가 부족한 경우 **Mflux Optimizations** 노드를 사용하세요. 생성을 위해 `low_ram`을 활성화하고 큰 이미지 디코딩을 위해 `vae_tiling`을 활성화하세요.
- **LoRA 호환성**: LoRA는 현재 기본 모델을 `quantize=8` (또는 None)로 로드해야 합니다.
- **해상도**: 너비와 높이는 16의 배수여야 합니다 (필요한 경우 자동으로 조정됨).
- **가이던스 (Guidance)**:
  - `dev` 모델은 guidance를 따릅니다 (기본값 ~3.5).
  - `schnell` 모델은 guidance를 무시합니다 (그대로 두어도 무방).
- **경로**:
  - 양자화된 모델: `ComfyUI/models/Mflux`
  - LoRA: `ComfyUI/models/loras` (정리를 위해 `Mflux` 하위 디렉토리 생성을 권장).
  - HuggingFace에서 자동 다운로드된 모델 (예: Z-Image Turbo 노드를 처음 사용할 때 `filipstrand/Z-Image-Turbo-mflux-4bit`): `User/.cache/huggingface/hub`에 위치하며, `Cmd + Shift + .`를 눌러 숨겨진 .cache 폴더를 볼 수 있습니다.

## 워크플로우 (Workflows)

`workflows` 폴더에서 JSON 예제를 확인하세요:
- `Mflux text2img.json`
- `Mflux img2img.json`
- `Mflux ControlNet.json`
- `Mflux Fill/Redux/Depth` 예제 (사용 가능한 경우)

Z-Image Turbo 워크플로우는 `examples` 폴더의 png 파일에 포함되어 있습니다:
- `Air_Z-Image-Turbo.png`
- `Air_Z-Image-Turbo_model_loader.png`
- `Air_Z-Image-Turbo_img2img_lora.png`

ComfyUI에서 노드가 빨간색으로 표시되면 Manager의 "Install Missing Custom Nodes" 기능을 사용하세요.

## 감사의 말 (Acknowledgements)

- **mflux**: [@filipstrand](https://github.com/filipstrand) 및 기여자분들.
- 초기 ComfyUI 통합 개념: **raysers**.
- MFlux-ComfyUI 2.0.0: **joonsoome**.
- 일부 코드 구조는 **MFLUX-WEBUI**에서 영감을 받았습니다.

## 라이선스 (License)

MIT
```