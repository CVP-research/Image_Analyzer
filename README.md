# Computer Vision Analysis Application

이미지에 대한 의미론적 분할(Semantic Segmentation)과 깊이 추정(Depth Estimation)을 수행하는 웹 기반 컴퓨터 비전 분석 도구입니다.

## 주요 기능

- **의미론적 분할**: OneFormer 모델을 사용하여 이미지의 객체와 영역을 자동으로 인식하고 분류
- **깊이 추정**: Depth Anything V2 모델을 사용하여 단일 이미지로부터 깊이 정보 추출
- **통합 분석**: 분할된 각 영역의 평균 깊이를 계산하여 표시
- **웹 인터페이스**: Gradio 기반의 사용하기 쉬운 대화형 웹 UI

## 프로젝트 구조

```
project/
├── server.py              # Gradio 웹 애플리케이션 메인 서버
├── segment.py             # 의미론적 분할 모듈
├── depth.py               # 깊이 추정 모듈
├── requirements.txt       # Python 의존성 패키지
├── checkpoints/           # 모델 체크포인트 디렉토리
│   └── depth_anything_v2_vits.pth
├── input/                 # 입력 이미지 디렉토리
│   └── input_images_here.md
├── segment/               # 분할 결과 출력 디렉토리 (자동 생성)
└── depth/                 # 깊이 맵 출력 디렉토리 (자동 생성)
```

## 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd project
```

### 2. 가상 환경 생성 및 활성화

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. 의존성 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. 모델 체크포인트 다운로드

Depth Anything V2 모델 체크포인트를 다운로드하여 `checkpoints/` 디렉토리에 저장합니다.

- [depth_anything_v2_vits.pth](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth) (약 100MB)

다른 크기의 모델을 사용하려면:
- `depth_anything_v2_vitb.pth` - Base 모델
- `depth_anything_v2_vitl.pth` - Large 모델
- `depth_anything_v2_vitg.pth` - Giant 모델

## 사용 방법

### 1. 이미지 준비

분석할 이미지를 `input/` 디렉토리에 저장합니다 (JPG, PNG 형식 지원).

### 2. 서버 실행

```bash
python server.py
```

### 3. 웹 인터페이스 접속

브라우저에서 다음 주소로 접속합니다:
```
http://localhost:8080
```

또는 콘솔에 표시되는 공개 URL을 사용하여 외부에서 접속할 수 있습니다.

### 4. 이미지 분석

- **Previous/Next 버튼**: 이미지 간 이동
- **분할 뷰**: 색상으로 구분된 객체 영역 및 레이블 표시
- **깊이 맵 뷰**: Spectral 컬러맵으로 시각화된 깊이 정보
- **JSON 레이블**: 각 분할 영역의 상세 정보 (ID, 레이블, 색상, 신뢰도)
- **깊이 통계**: 각 영역의 평균 깊이 (오름차순 정렬)

## 기술 스택

### 딥러닝 모델

- **OneFormer** (`shi-labs/oneformer_coco_swin_large`)
  - Swin Transformer 기반 범용 이미지 분할 모델
  - COCO 데이터셋으로 학습
  - GPU 가속 지원

- **Depth Anything V2** (ViT-Small)
  - Vision Transformer 기반 단안 깊이 추정 모델
  - 518px 입력 크기
  - CUDA/CPU 모드 자동 선택

### 프레임워크 및 라이브러리

- **Gradio**: 웹 UI 프레임워크
- **PyTorch**: 딥러닝 프레임워크
- **Transformers**: Hugging Face 모델 라이브러리
- **OpenCV**: 이미지 처리
- **NumPy**: 수치 연산
- **Matplotlib**: 시각화 (컬러맵)
- **Pillow**: 이미지 입출력

## 주요 파일 설명

### [server.py](server.py)
메인 애플리케이션 서버로 다음 기능을 담당합니다:
- Gradio 웹 인터페이스 구성
- 이미지 로딩 및 캐싱 시스템
- 분할 및 깊이 분석 파이프라인 조율
- 영역별 평균 깊이 계산
- 백그라운드 이미지 사전 로딩

### [segment.py](segment.py)
의미론적 분할 모듈:
- OneFormer 모델 초기화 및 추론
- 100개의 고유 색상 생성
- 분할 결과 오버레이 생성
- JSON 형식의 레이블 정보 출력

### [depth.py](depth.py)
깊이 추정 모듈:
- Depth Anything V2 모델 초기화
- 단일 이미지 깊이 추론
- Spectral 컬러맵 시각화
- 원시 깊이 배열 반환

## 캐싱 시스템

성능 최적화를 위해 결과를 캐싱합니다:

- **분할 결과**: `segment/<filename>.png`, `segment/<filename>_labels.pkl`
- **깊이 결과**: `depth/<filename>.png`, `depth/<filename>.npy`

캐시된 결과는 재사용되어 반복 조회 시 처리 시간을 크게 단축합니다.

## 서버 설정

기본 설정:
- **호스트**: `0.0.0.0` (모든 네트워크 인터페이스)
- **포트**: `8080`
- **공유**: 활성화 (Gradio 공개 URL 생성)

[server.py](server.py)에서 설정을 변경할 수 있습니다.

## 시스템 요구사항

- **Python**: 3.8 이상
- **GPU**: CUDA 지원 GPU 권장 (CPU 모드도 지원하지만 느림)
- **메모리**: 최소 8GB RAM
- **저장공간**: 모델 체크포인트 및 캐시를 위한 충분한 공간

## 문제 해결

### GPU 메모리 부족

더 작은 모델을 사용하거나 [depth.py:24](depth.py#L24)에서 `encoder` 파라미터를 변경:
```python
encoder = 'vits'  # 또는 'vitb', 'vitl', 'vitg'
```

### 포트 이미 사용 중

[server.py:240](server.py#L240)에서 포트 번호를 변경:
```python
demo.launch(server_name="0.0.0.0", server_port=다른포트번호, share=True)
```

### 모델 다운로드 실패

체크포인트 파일을 수동으로 다운로드하여 `checkpoints/` 디렉토리에 저장하세요.

## 라이선스

이 프로젝트는 다음 오픈소스 모델을 사용합니다:
- OneFormer: [SHI Labs GitHub](https://github.com/SHI-Labs/OneFormer)
- Depth Anything V2: [Depth-Anything-V2 GitHub](https://github.com/DepthAnything/Depth-Anything-V2)

각 모델의 라이선스를 확인하세요.

## 기여

버그 리포트나 기능 제안은 Issues를 통해 제출해주세요.

## 참고 자료

- [Gradio 문서](https://gradio.app/docs/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Depth Anything V2 논문](https://arxiv.org/abs/2406.09414)
- [OneFormer 논문](https://arxiv.org/abs/2211.06220)
