"""
Image-to-Object Pipeline

완전한 파이프라인 흐름:
1. input/ 폴더의 이미지를 SAM으로 누끼 따기 (segment_input.py 서버)
2. 누끼 딴 이미지들에 대해 포즈 추정 (pose_estimator.py 서버)
3. 포즈 기반으로 최적의 3장 선택 (ViewSelector)
4. 선택된 3장으로 Veo3 360도 영상 생성 (VideoGeneratorFacade)
5. 생성된 영상에서 객체 분리 (VideoObjectExtractor)
5.5. 추출된 객체 검수 - 사용자가 웹 UI에서 승인/버리기 (review_objects.py 서버)
6. 모든 객체를 모아 YOLO 파인튜닝용 데이터셋 생성 (DatasetBuilder)

사용법:
    python main.py {프로젝트명}
    python main.py my_project --debug
    python main.py my_project --mock   # Veo3 스킵, 기존 영상 사용
    
디렉토리 구조:
    data/{project_name}/
    ├── input/                    # 여기에 이미지를 넣으세요
    │   └── pose/camera_poses.json
    ├── input_masked/             # 누끼 딴 결과
    └── output/
        ├── video/                # Veo3 생성 영상 (mock 시 여기에 영상 넣기)
        ├── masked_frames/        # 영상에서 추출한 객체 (검수 전)
        ├── dataset/              # 기타 출력물
        └── yolo_dataset/         # YOLO 학습용 데이터셋
            ├── images/train/
            ├── images/val/
            ├── labels/train/
            └── labels/val/
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path
import time
from typing import List, Optional

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from util.utils import DebugMode
from util.paths import JobPaths


class Pipeline:
    """
    이미지 → 비디오 → 객체 분할 파이프라인 오케스트레이터
    
    파이프라인 단계:
    1. Segmentation Server: input 이미지 누끼 따기
    2. Pose Estimator Server: 카메라 포즈 추정
    3. View Selection: 최적 3장 선택
    4. Video Generation: Veo3로 360도 영상 생성
    5. Object Extraction: 영상에서 객체 분리 저장
    """
    
    def __init__(self, project_name: str, debug: bool = False, mock_veo3: bool = False):
        self.debug_mode = DebugMode(debug)
        self.mock_veo3 = mock_veo3
        self.base_dir = Path(__file__).parent
        
        # JobPaths로 경로 관리
        self.paths = JobPaths(base_dir=self.base_dir, project_name=project_name)
        
        print("=" * 60)
        print(f"📦 프로젝트: {project_name}")
        print("=" * 60)
        
        if self.debug_mode.enabled:
            print("DEBUG MODE ENABLED - API 호출 스킵됨")
        if self.mock_veo3:
            print("MOCK MODE ENABLED - Veo3 스킵, 기존 영상 사용")
        if self.debug_mode.enabled or self.mock_veo3:
            print("=" * 60)

        # 서버 스크립트 경로
        self.segment_server_script = self.base_dir / "servers" / "segment_input.py"
        self.pose_estimator_script = self.base_dir / "servers" / "pose_estimator.py"
        self.review_server_script = self.base_dir / "servers" / "review_objects.py"
        
        # SAM 모델 경로
        self.sam_model_path = self.base_dir / "sam2_l.pt"
        
        # Google 인증 정보 경로
        self.credentials_path = self.base_dir / "credentials.json"
        
        # 디렉토리 생성 및 경로 출력
        self.paths.create_dirs()
        self.paths.print_paths()

    def _check_input_images(self) -> List[Path]:
        """input 폴더에 이미지가 있는지 확인"""
        if not self.paths.input_dir.exists():
            print(f"\n❌ 오류: 프로젝트 폴더가 없습니다: {self.paths.input_dir}")
            print(f"   data/{self.paths.project_name}/input/ 폴더를 만들고 이미지를 넣으세요.")
            self._list_available_projects()
            sys.exit(1)
            
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        images = [
            p for p in self.paths.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in image_extensions
        ]
        if not images:
            print(f"\n❌ 오류: {self.paths.input_dir}에 이미지 파일이 없습니다.")
            print(f"   {self.paths.project_name}/input/ 폴더에 이미지를 넣으세요.")
            sys.exit(1)
        print(f"\n✓ {len(images)}개의 입력 이미지 발견")
        return sorted(images)
    
    def _list_available_projects(self):
        """사용 가능한 프로젝트 목록 출력"""
        projects = JobPaths.list_projects(self.base_dir)
        if projects:
            print(f"\n📂 사용 가능한 프로젝트:")
            for p in projects:
                print(f"   - {p}")
        else:
            print(f"\n   data/ 폴더 안에 프로젝트가 없습니다.")
            print(f"   data/{{프로젝트명}}/input/ 폴더를 만들고 이미지를 넣으세요.")

    def _run_server(self, script_path: Path, server_name: str, port: int = 8000, env_vars: dict = None) -> subprocess.Popen:
        """서버 스크립트를 subprocess로 실행"""
        if not script_path.exists():
            raise FileNotFoundError(f"{server_name} 스크립트를 찾을 수 없습니다: {script_path}")

        print(f"\n--- {server_name} 시작 ---")
        
        # 환경 변수 설정
        import os
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
        
        # 직접 Python 스크립트 실행 (uvicorn이 내장됨)
        command = [sys.executable, str(script_path)]
        
        process = subprocess.Popen(
            command,
            cwd=str(script_path.parent),
            env=env
        )
        
        print(f"✓ {server_name} 시작됨 (PID: {process.pid}, 포트: {port})")
        print(f"   브라우저에서 열기: http://localhost:{port}")
        
        # 서버 시작 대기
        time.sleep(3)
        return process

    def _wait_for_server(self, server_process: subprocess.Popen, server_name: str):
        """서버가 스스로 종료될 때까지 대기 (웹 UI에서 완료 버튼 클릭 시 종료됨)"""
        print(f"\n>>> {server_name} 작업 중... (웹 UI에서 '작업 완료' 버튼을 누르면 자동으로 다음 단계로 진행)")
        
        try:
            # 서버가 스스로 종료될 때까지 대기
            return_code = server_process.wait()
            print(f"✓ {server_name} 정상 종료됨 (exit code: {return_code})")
        except KeyboardInterrupt:
            print(f"\n--- {server_name} 수동 종료 중... ---")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
            raise

    # ==========================================
    # STEP 1: 이미지 세그멘테이션 (누끼 따기)
    # ==========================================
    def _run_segmentation(self):
        """Step 1: SAM 기반 이미지 세그멘테이션 서버 실행"""
        print("\n" + "=" * 60)
        print("STEP 1: 이미지 세그멘테이션 (누끼 따기)")
        print("=" * 60)
        
        # 환경 변수로 경로 전달
        env_vars = {
            "SEGMENT_INPUT_DIR": str(self.paths.input_dir),
            "SEGMENT_OUTPUT_DIR": str(self.paths.input_masked_dir),
        }
        
        server_process = self._run_server(
            self.segment_server_script, 
            "Segmentation Server",
            port=8001,
            env_vars=env_vars
        )
        self._wait_for_server(server_process, "Segmentation Server")
        
        # 결과 확인
        segmented_images = list(self.paths.input_masked_dir.glob("*.png"))
        print(f"✓ 세그멘테이션 완료: {len(segmented_images)}개 이미지 저장됨")

    # ==========================================
    # STEP 2: 포즈 추정
    # ==========================================
    def _run_pose_estimation(self):
        """Step 2: 카메라 포즈 추정 서버 실행"""
        print("\n" + "=" * 60)
        print("STEP 2: 카메라 포즈 추정")
        print("=" * 60)
        
        # 환경 변수로 경로 전달
        env_vars = {
            "POSE_INPUT_DIR": str(self.paths.input_masked_dir),
            "POSE_OUTPUT_DIR": str(self.paths.pose_dir),
        }
        
        server_process = self._run_server(
            self.pose_estimator_script, 
            "Pose Estimator Server",
            port=8002,
            env_vars=env_vars
        )
        self._wait_for_server(server_process, "Pose Estimator Server")
        
        # 결과 확인
        if self.paths.poses_file.exists():
            print(f"✓ 포즈 추정 완료: {self.paths.poses_file}")
        else:
            print("⚠ 경고: 포즈 파일이 생성되지 않았습니다. View Selection에서 랜덤 선택으로 대체됩니다.")

    # ==========================================
    # STEP 3: 최적 뷰 선택 (3장)
    # ==========================================
    def _select_views(self) -> List[Path]:
        """Step 3: 포즈 기반 최적 3장 선택"""
        print("\n" + "=" * 60)
        print("STEP 3: 최적 뷰 선택 (3장)")
        print("=" * 60)
        
        from data.image.view_selector import ViewSelector
        
        # 세그멘테이션된 이미지 목록
        input_images = list(self.paths.input_masked_dir.glob("*.png"))
        if not input_images:
            # fallback: 원본 input 폴더 사용
            input_images = self._check_input_images()
        
        # ViewSelector로 최적 3장 선택
        selector = ViewSelector(
            input_images=input_images,
            poses_file=self.paths.poses_file if self.paths.poses_file.exists() else None
        )
        selected_views = selector.select_best_views(num_views=3)
        
        print(f"✓ 뷰 선택 완료: {len(selected_views)}장")
        return selected_views

    # ==========================================
    # STEP 4: Veo3 360도 영상 생성
    # ==========================================
    def _generate_video(self, selected_views: List[Path]) -> Optional[Path]:
        """Step 4: Veo3 API로 360도 영상 생성 (또는 mock 모드에서 기존 영상 사용)"""
        print("\n" + "=" * 60)
        print("STEP 4: Veo3 360도 영상 생성")
        print("=" * 60)
        
        # Mock 모드: 기존 영상 파일 사용
        if self.mock_veo3:
            print("🎭 Mock 모드: Veo3 API 스킵, 기존 영상 파일 검색")
            
            # video_dir에서 영상 파일 찾기
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']
            video_files = []
            for ext in video_extensions:
                video_files.extend(list(self.paths.video_dir.glob(ext)))
            
            if video_files:
                video_path = video_files[0]  # 첫 번째 영상 사용
                print(f"✓ 기존 영상 발견: {video_path}")
                return video_path
            else:
                print(f"⚠ 영상 파일이 없습니다. 아래 경로에 영상을 넣어주세요:")
                print(f"   {self.paths.video_dir}/")
                return None
        
        from data.video.facade import VideoGeneratorFacade
        from data.video.api.veo3 import Veo3VideoGenerator
        
        # Veo3 생성기 등록
        VideoGeneratorFacade.register_generator("veo3", Veo3VideoGenerator)
        
        # Facade 생성
        facade = VideoGeneratorFacade(
            generator_name="veo3",
            video_dir=self.paths.video_dir,
            debug_mode=self.debug_mode,
            credentials_path=self.credentials_path,
            project="your-gcp-project",  # TODO: 실제 프로젝트 ID로 변경 필요
            location="us-central1"
        )
        
        # 영상 생성
        try:
            video_path = facade.generate_video(
                selected_images=selected_views,
                additional_prompt="Focus on the object, smooth rotation."
            )
            print(f"✓ 영상 생성 완료: {video_path}")
            return video_path
        except Exception as e:
            print(f"❌ 영상 생성 실패: {e}")
            return None

    # ==========================================
    # STEP 5: 영상에서 객체 분리
    # ==========================================
    def _extract_objects(self, video_path: Optional[Path]):
        """Step 5: 생성된 영상에서 객체 분리 (mock 모드에서는 스킵)"""
        print("\n" + "=" * 60)
        print("STEP 5: 영상에서 객체 분리")
        print("=" * 60)
        
        # Mock 모드: 사용자가 masked_frames에 이미지를 미리 넣어둠
        if self.mock_veo3:
            existing_objects = list(self.paths.masked_frames_dir.glob("*.png"))
            if existing_objects:
                print(f"🎭 Mock 모드: 기존 객체 이미지 {len(existing_objects)}개 발견")
                print(f"   위치: {self.paths.masked_frames_dir}")
                return
            else:
                print(f"⚠ Mock 모드이지만 객체 이미지가 없습니다.")
                print(f"   아래 경로에 누끼 딴 PNG 이미지를 넣어주세요:")
                print(f"   {self.paths.masked_frames_dir}/")
                return
        
        if video_path is None or not video_path.exists():
            print("⚠ 영상 파일이 없어 객체 분리를 건너뜁니다.")
            return
        
        from data.object.object_extractor import VideoObjectExtractor
        
        extractor = VideoObjectExtractor(
            input_video_path=video_path,
            extracted_frames_dir=self.paths.video_frames_dir,
            masked_objects_dir=self.paths.masked_frames_dir,
            sam_model_path=self.sam_model_path
        )
        
        extractor.run_pipeline()
        
        # 결과 확인
        extracted_count = len(list(self.paths.masked_frames_dir.glob("*.png")))
        print(f"✓ 객체 분리 완료: {extracted_count}개 객체 저장됨")

    # ==========================================
    # STEP 5.5: 객체 검수 (사용자 확인)
    # ==========================================
    def _review_objects(self):
        """Step 5.5: 추출된 객체를 사용자가 검수"""
        print("\n" + "=" * 60)
        print("STEP 5.5: 객체 검수")
        print("=" * 60)
        
        # 검수할 객체가 있는지 확인
        objects_to_review = list(self.paths.masked_frames_dir.glob("*.png"))
        if not objects_to_review:
            print("⚠ 검수할 객체가 없습니다. 검수 단계를 건너뜁니다.")
            return
        
        print(f"✓ {len(objects_to_review)}개 객체 검수 대기")
        print("  (버리는 이미지만 삭제됩니다)")
        
        # 환경 변수로 경로 전달
        env_vars = {
            "REVIEW_OBJECTS_DIR": str(self.paths.masked_frames_dir),
        }
        
        server_process = self._run_server(
            self.review_server_script,
            "Review Server",
            port=8003,
            env_vars=env_vars
        )
        self._wait_for_server(server_process, "Review Server")
        
        # 결과 확인 (삭제 후 남은 객체 수)
        remaining_count = len(list(self.paths.masked_frames_dir.glob("*.png")))
        print(f"✓ 검수 완료: {remaining_count}개 객체 남음")

    # ==========================================
    # STEP 6: YOLO 데이터셋 생성
    # ==========================================
    def _generate_dataset(self):
        """Step 6: YOLO 파인튜닝용 데이터셋 생성"""
        print("\n" + "=" * 60)
        print("STEP 6: YOLO 데이터셋 생성")
        print("=" * 60)
        
        from data.dataset_builder import DatasetBuilder, DatasetConfig
        
        # 객체 소스 디렉토리 확인
        obj_dirs = self.paths.get_object_source_dirs()
        valid_obj_dirs = [d for d in obj_dirs if Path(d).exists() and list(Path(d).glob("*.png"))]
        
        if not valid_obj_dirs:
            print("⚠ 객체 이미지가 없어 데이터셋 생성을 건너뜁니다.")
            print(f"   확인한 경로: {obj_dirs}")
            return
        
        # 배경 데이터셋 확인
        if not self.paths.bg_dataset_dir.exists():
            print(f"⚠ 배경 데이터셋 폴더가 없습니다: {self.paths.bg_dataset_dir}")
            print("   dataset/ 폴더에 Places365 등의 배경 이미지를 넣으세요.")
            return
        
        # 객체 수 확인
        total_objects = sum(len(list(Path(d).glob("*.png"))) for d in valid_obj_dirs)
        print(f"✓ 발견된 객체 이미지: {total_objects}개")
        print(f"  - 소스 폴더: {valid_obj_dirs}")
        
        # 항상 3000개의 positive 샘플 생성 (객체를 랜덤 선택하여 반복 사용)
        target_positive = 3000
        avg_copies = target_positive / total_objects
        num_bg_groups = target_positive
        num_negative_samples = num_bg_groups // 5  # 600개
        
        print(f"✓ 객체당 평균 {avg_copies:.1f}회 사용 → 총 {num_bg_groups}개 positive, {num_negative_samples}개 negative")
        
        # DatasetConfig 생성
        config = DatasetConfig(
            bg_root=str(self.paths.bg_dataset_dir),
            obj_root=valid_obj_dirs,
            output_root=str(self.paths.yolo_dataset_dir),
            target_size=1024,
            tile_size=256,
            num_bg_groups=num_bg_groups,
            num_negative_samples=num_negative_samples,
            val_split_ratio=0.2,
            class_id=0,
            device=0
        )
        
        # 데이터셋 생성
        try:
            builder = DatasetBuilder(config)
            builder.generate_datasets()
            
            # 결과 확인
            train_count = len(list((self.paths.yolo_dataset_dir / "images" / "train").glob("*.jpg")))
            val_count = len(list((self.paths.yolo_dataset_dir / "images" / "val").glob("*.jpg")))
            print(f"✓ 데이터셋 생성 완료: train={train_count}, val={val_count}")
            print(f"  저장 위치: {self.paths.yolo_dataset_dir}")
        except Exception as e:
            print(f"❌ 데이터셋 생성 실패: {e}")
            import traceback
            traceback.print_exc()

    # ==========================================
    # 전체 파이프라인 실행
    # ==========================================
    def run(self):
        """전체 파이프라인 실행"""
        print("\n" + "=" * 60)
        print("🚀 Image-to-Object 파이프라인 시작")
        print("=" * 60)
        
        start_time = time.time()
        
        # 입력 이미지 확인
        self._check_input_images()
        
        # Step 1: 세그멘테이션
        self._run_segmentation()
        
        # Step 2: 포즈 추정
        self._run_pose_estimation()
        
        # Step 3: 뷰 선택
        selected_views = self._select_views()
        
        # Step 4: 영상 생성
        video_path = self._generate_video(selected_views)
        
        # Step 5: 객체 분리
        self._extract_objects(video_path)
        
        # Step 5.5: 객체 검수 (사용자 확인)
        self._review_objects()
        
        # Step 6: YOLO 데이터셋 생성
        self._generate_dataset()
        
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"✅ 파이프라인 완료! (소요 시간: {elapsed:.1f}초)")
        print("=" * 60)
        print(f"\n📁 결과물 위치:")
        print(f"   - 누끼 결과:     {self.paths.input_masked_dir}")
        print(f"   - 생성된 영상:   {self.paths.video_dir}")
        print(f"   - 분리된 객체:   {self.paths.masked_frames_dir}")
        print(f"   - YOLO 데이터셋: {self.paths.yolo_dataset_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image-to-Object 파이프라인: 이미지 → 누끼 → 포즈 → 뷰 선택 → Veo3 → 객체 분리",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
    python main.py my_project          # my_project 프로젝트 실행
    python main.py my_project --debug  # 디버그 모드 (API 호출 스킵)
    python main.py my_project --mock   # Veo3 스킵, 기존 영상 사용
    
디렉토리 구조:
    data/{project}/input/     <- 여기에 이미지를 넣으세요
    data/{project}/output/video/ <- mock 모드시 영상을 여기에
    data/{project}/output/    <- 결과물이 여기에 저장됩니다
        """
    )
    parser.add_argument(
        "project",
        type=str,
        help="프로젝트명 (data/{project}/input/ 폴더에서 이미지를 읽음)"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="디버그 모드 활성화 (API 호출 스킵)"
    )
    parser.add_argument(
        "--mock", "-m",
        action="store_true",
        help="Mock 모드: Veo3 API 스킵하고 기존 영상 파일 사용"
    )
    args = parser.parse_args()

    pipeline = Pipeline(project_name=args.project, debug=args.debug, mock_veo3=args.mock)
    pipeline.run()
