# pipeline/__init__.py
"""
파이프라인 패키지

- run: 메인 진입점
- file_patterns: 파일명 패턴 정의
- generate_dataset: YOLO 데이터셋 생성 (다중 폴더 지원, harmonization 옵션)
- prepare_backgrounds: 배경 준비 (타일링/복사 모드)
- veo3_pipeline: Veo3 멀티뷰 파이프라인
- extract_objects: 객체 추출
- view_selector: 최적 뷰 선택
- reorganize_dataset: 데이터 재분배
"""

from pipeline.file_patterns import FilePatterns
from pipeline.generate_dataset import DatasetConfig, DatasetBuilder
from pipeline.prepare_backgrounds import BackgroundConfig
