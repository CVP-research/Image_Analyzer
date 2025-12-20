# Image Analyzer

Few-shot 학습을 위한 합성 데이터셋 생성 파이프라인입니다. 소수의 객체 이미지로부터 다양한 배경에 자연스럽게 합성된 YOLO 학습용 데이터셋을 자동 생성합니다.

## Features

- **Veo3 Multi-view Generation**: 입력 이미지에서 360도 회전 영상 생성
- **Semantic Background Matching**: 객체와 의미적으로 어울리는 배경 자동 선택
- **Depth-aware Compositing**: 깊이 정보를 활용한 자연스러운 합성
- **Image Harmonization**: libcom을 활용한 조명/색상 조화 (선택적)
- **YOLO Dataset Generation**: 학습용 데이터셋 자동 생성 및 분할

## Project Structure

```
Image_Analyzer/
├── modules/           # 핵심 모듈 (depth, segment, composite 등)
├── pipeline/          # 파이프라인 스크립트
│   ├── run.py                    # 메인 진입점
│   ├── file_patterns.py          # 파일명 패턴 관리
│   ├── generate_dataset.py       # YOLO 데이터셋 생성
│   ├── prepare_backgrounds.py    # 배경 이미지 준비
│   ├── pose_estimator_ui.py      # 카메라 포즈 등록 UI
│   ├── veo3_pipeline.py          # Veo3 파이프라인
│   ├── extract_objects.py        # 객체 추출
│   ├── view_selector.py          # 최적 뷰 선택
│   └── jobs/                     # Job별 데이터 저장
├── finetune/          # YOLO 학습 및 추론
├── metric/            # 평가 및 시각화
├── requirements.txt   # 의존성 목록
└── setup.sh           # 초기 설정 스크립트
```

## Installation

### 1. 저장소 클론 및 서브모듈 초기화

```bash
git clone https://github.com/your-repo/Image_Analyzer.git
cd Image_Analyzer
./setup.sh
```

### 2. 가상환경 생성 및 의존성 설치

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. PyTorch 설치 (시스템에 맞게)

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision

# macOS (Apple Silicon)
pip install torch torchvision
```

### 4. Google Cloud 인증 설정 (Veo3 사용 시)

```bash
# GCP 서비스 계정 키 파일을 프로젝트 루트에 저장
cp /path/to/your/credentials.json ./credentials.json
```

## Quick Start

### 기본 실행

```bash
python pipeline/run.py
```

실행 시:
1. 기존 job 선택 또는 새 job 생성
2. `input/` 폴더에 객체 이미지 배치
3. 카메라 포즈 등록 (Gradio UI)
4. 자동으로 데이터셋 생성

### 디버그 모드

무거운 모델 로딩 없이 파이프라인 흐름만 테스트:

```bash
# 디버그 모드 (모델 로딩, API 호출 스킵)
python pipeline/run.py --debug

# 디버그 + 새 job 자동 생성
python pipeline/run.py --debug --new

# 특정 job 지정
python pipeline/run.py --debug --job 20231220_143052
```

디버그 모드에서 스킵되는 작업:
- GPU 모델 로딩 (SAM, YOLO, Depth-Anything 등)
- Google Veo3 API 호출
- Semantic matching (sentence-transformers)
- 실제 합성 연산

디버그 모드에서 자동 생성되는 더미 데이터:
- 입력 이미지 (input 폴더가 비어있을 때)
- 카메라 포즈 JSON
- 프레임 및 마스크 이미지
- 배경 이미지

### 도움말

```bash
python pipeline/run.py --help
```

```
usage: run.py [-h] [--debug] [--new] [--job JOB]

Few-shot Semantic Compositing Pipeline

options:
  -h, --help   show this help message and exit
  --debug, -d  디버그 모드: 모델 로딩, API 호출, GPU 연산 스킵
  --new, -n    새 job 자동 생성 (선택 프롬프트 스킵)
  --job JOB    특정 job 이름 지정 (예: 20231220_143052)

Examples:
  python pipeline/run.py                 # 일반 실행
  python pipeline/run.py --debug         # 디버그 모드
  python pipeline/run.py --debug --new   # 디버그 + 새 job 자동 생성
```

## Pipeline Steps

```
Step 1:   입력 이미지 로드
Step 1.5: 카메라 포즈 등록 (Gradio UI)
Step 1.8: 최적 뷰 선택
Step 2:   Veo3로 360도 영상 생성
Step 2.5: 영상에서 프레임 추출
Step 3:   객체 Segmentation (SAM)
Step 4:   의미적으로 적합한 배경 검색
Step 4.5: 객체 로드
Step 5:   Depth-aware 합성 및 YOLO 라벨 생성
```

## Job 시스템

각 실행은 타임스탬프 기반의 job 폴더에 저장됩니다:

```
pipeline/jobs/
└── 20231220_143052/          # Job 폴더 (타임스탬프)
    ├── input/                # 입력 이미지
    │   └── pose/             # 카메라 포즈 데이터
    │       └── camera_poses.json
    └── output/
        ├── video/            # Veo3 생성 영상
        ├── frames/           # 추출된 프레임
        ├── masked_frames/    # Segmentation 결과
        └── dataset/          # 최종 YOLO 데이터셋
            └── result/
                ├── images/
                │   ├── train/
                │   └── val/
                └── labels/
                    ├── train/
                    └── val/
```

## 개별 모듈 사용

### 데이터셋 생성

```python
from pipeline.generate_dataset import DatasetConfig, DatasetBuilder, run

# 설정
config = DatasetConfig(
    bg_root="./backgrounds",
    obj_root=["./objects/view1", "./objects/view2"],  # 다중 폴더 지원
    output_root="./dataset/train",
    num_bg_groups=3000,
    num_negative_samples=2000,
    use_harmonization=True,  # libcom 필요
)

# 실행
run(config)
```

### 배경 준비

```python
from pipeline.prepare_backgrounds import BackgroundConfig, run

# 타일링 모드 (4x4 = 1024x1024)
config = BackgroundConfig(
    source_dir="./places365/val",
    dest_dir="./backgrounds",
    target_rgb=(160, 110, 60),  # 갈색 계열 필터
    top_n=32000,
    use_tiling=True,
    tile_size=256,
    grid_size=4,
)

run(config)
```

### 파일명 패턴

```python
from pipeline.file_patterns import FilePatterns

# 패턴 사용
video_name = FilePatterns.VIDEO                    # "veo3_360.mp4"
frame_name = FilePatterns.frame_name(0)            # "frame_0000.png"
masked_name = FilePatterns.masked_name("frame_0000")  # "frame_0000_masked.png"
composite_name = FilePatterns.composite_name("train", 0, 5)  # "composite_train_bg0000_obj0005.png"
```

## 의존성

### 필수

```
gradio
transformers
pillow
numpy
opencv-python
timm
scipy
ultralytics
pyyaml
tqdm
scikit-image
sentence-transformers
torch
torchvision
```

### 선택적

```
libcom          # Image harmonization
realesrgan      # Image upscaling
google-genai    # Veo3 API
```

## YOLO 학습

생성된 데이터셋으로 YOLO 모델 학습:

```bash
cd finetune
python train.py --data ../pipeline/jobs/{JOB_NAME}/output/dataset/result/data.yaml
```

## Troubleshooting

### 의존성 오류

```bash
# 전체 의존성 재설치
pip install -r requirements.txt

# 디버그 모드로 테스트
python pipeline/run.py --debug
```

### GPU 메모리 부족

```python
# pipeline/run.py에서 배치 크기 조정
max_workers=2  # 기본값 5에서 줄임
```

### Veo3 API 오류

1. `credentials.json` 파일 확인
2. GCP 프로젝트 권한 확인
3. Vertex AI API 활성화 확인

## License

MIT License - see [LICENSE](LICENSE)
