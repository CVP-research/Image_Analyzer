# pipeline/file_patterns.py
"""
파일명 패턴 정의 (각 Step의 Input/Output 연결)

각 파이프라인 Step의 출력 파일명이 다음 Step의 입력 파일명으로 연결됩니다.
파일명 패턴을 변경하려면 이 파일만 수정하면 됩니다.
"""


class FilePatterns:
    """각 Step의 파일명 패턴을 중앙 관리"""

    # =========================================================================
    # Step 2: Veo3 영상 생성
    # =========================================================================
    VIDEO = "veo3_360.mp4"

    # =========================================================================
    # Step 2.5: 프레임 추출
    # Input: VIDEO
    # Output: frame_0001.png, frame_0002.png, ...
    # =========================================================================
    FRAME = "frame_{idx:04d}.png"
    FRAME_GLOB = "frame_*.png"

    # =========================================================================
    # Step 3: 객체 세그멘테이션
    # Input: FRAME_GLOB
    # Output: frame_0001_masked.png, frame_0002_masked.png, ...
    # =========================================================================
    MASKED = "{stem}_masked.png"
    MASKED_GLOB = "*_masked.png"

    # =========================================================================
    # Step 5: 합성 결과
    # Input: MASKED_GLOB + backgrounds
    # Output: composite_train_bg0001_obj0002.png, ...
    # =========================================================================
    COMPOSITE = "composite_{split}_bg{bg_idx:04d}_obj{obj_idx:04d}.png"
    COMPOSITE_GLOB = "composite_*.png"

    # =========================================================================
    # 배경 이미지 (negative sampling용)
    # =========================================================================
    BACKGROUND = "bg{idx:04d}.png"
    BACKGROUND_GLOB = "bg*.png"

    # =========================================================================
    # Helper Methods
    # =========================================================================
    @classmethod
    def frame_name(cls, idx: int) -> str:
        """프레임 파일명 생성 (예: frame_0001.png)"""
        return cls.FRAME.format(idx=idx)

    @classmethod
    def masked_name(cls, frame_stem: str) -> str:
        """마스크 파일명 생성 (예: frame_0001_masked.png)"""
        return cls.MASKED.format(stem=frame_stem)

    @classmethod
    def composite_name(cls, split: str, bg_idx: int, obj_idx: int) -> str:
        """합성 이미지 파일명 생성 (예: composite_train_bg0001_obj0002.png)"""
        return cls.COMPOSITE.format(split=split, bg_idx=bg_idx, obj_idx=obj_idx)

    @classmethod
    def background_name(cls, idx: int) -> str:
        """배경 이미지 파일명 생성 (예: bg0001.png)"""
        return cls.BACKGROUND.format(idx=idx)
