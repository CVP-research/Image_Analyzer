from pathlib import Path


class JobPaths:
    """
    프로젝트 기반 경로 관리 클래스
    
    구조:
    {base_dir}/{project_name}/
    ├── input/                    # 입력 이미지
    │   └── pose/                 # 포즈 데이터
    │       └── camera_poses.json
    ├── input_masked/             # 누끼 딴 이미지 (세그멘테이션 결과)
    └── output/
        ├── video/                # Veo3 생성 영상
        ├── masked_frames/        # 영상에서 추출한 객체
        └── dataset/              # 최종 결과물
    """
    
    def __init__(self, base_dir: Path, project_name: str):
        self._base_dir = base_dir.resolve()
        self._project_name = project_name
        self._job_dir = self._base_dir / project_name
        self._setup_paths()

    def _setup_paths(self):
        # 내부 변수
        self._input_dir = self._job_dir / "input"
        self._input_masked_dir = self._job_dir / "input_masked"
        self._video_dir = self._job_dir / "output" / "video"
        self._video_frames_dir = self._job_dir / "output" / "video_frames"
        self._masked_frames_dir = self._job_dir / "output" / "masked_frames"
        self._output_dir = self._job_dir / "output" / "dataset"
        self._pose_dir = self._input_dir / "pose"
        self._poses_file = self._pose_dir / "camera_poses.json"

    def create_dirs(self):
        """모든 디렉토리 생성"""
        for dir_path in [
            self._input_dir,
            self._input_masked_dir,
            self._video_dir,
            self._video_frames_dir,
            self._output_dir,
            self._masked_frames_dir,
            self._pose_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def print_paths(self):
        print(f"\n[📂 프로젝트: {self._project_name}]")
        print(f"  Input:          {self._input_dir}")
        print(f"  Input Masked:   {self._input_masked_dir}")
        print(f"  Video Output:   {self._video_dir}")
        print(f"  Video Frames:   {self._video_frames_dir}")
        print(f"  Masked Frames:  {self._masked_frames_dir}")
        print(f"  Dataset Output: {self._output_dir}")
        print(f"  Pose Dir:       {self._pose_dir}")

    # ------------------ Properties ------------------
    @property
    def project_name(self) -> str:
        return self._project_name
    
    @property
    def base_dir(self) -> Path:
        return self._base_dir
    
    @property
    def job_dir(self) -> Path:
        return self._job_dir

    @property
    def input_dir(self) -> Path:
        return self._input_dir

    @property
    def input_masked_dir(self) -> Path:
        return self._input_masked_dir

    @property
    def video_dir(self) -> Path:
        return self._video_dir

    @property
    def video_frames_dir(self) -> Path:
        return self._video_frames_dir

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def masked_frames_dir(self) -> Path:
        return self._masked_frames_dir

    @property
    def pose_dir(self) -> Path:
        return self._pose_dir

    @property
    def poses_file(self) -> Path:
        return self._poses_file

    @property
    def bg_dataset_dir(self) -> Path:
        """배경 이미지 데이터셋 디렉토리 (Places365 등)"""
        return self._base_dir / "dataset"

    @property
    def yolo_dataset_dir(self) -> Path:
        """YOLO 학습용 데이터셋 출력 디렉토리"""
        return self._output_dir / "yolo_dataset"

    def get_object_source_dirs(self) -> list:
        """
        객체 소스 디렉토리 목록 반환
        - input_masked: 원본 이미지에서 누끼 딴 것
        - masked_frames: Veo3 영상에서 추출 후 검수 통과한 것
        """
        return [
            str(self._input_masked_dir),
            str(self._masked_frames_dir)
        ]

    # ------------------ 편리 메소드 ------------------
    def get_video_file_path(self, filename: str) -> Path:
        return self.video_dir / filename

    def get_output_file_path(self, filename: str) -> Path:
        return self.output_dir / filename

    def get_masked_frame_path(self, filename: str) -> Path:
        return self.masked_frames_dir / filename
    
    @classmethod
    def list_projects(cls, base_dir: Path) -> list:
        """사용 가능한 프로젝트 목록 반환"""
        data_dir = base_dir / "data"
        if not data_dir.exists():
            return []
        return sorted([
            d.name for d in data_dir.iterdir() 
            if d.is_dir() and not d.name.startswith('.') and (d / "input").exists()
        ])
