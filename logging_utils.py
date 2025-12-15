"""
Logging utilities for the pipeline
"""

import logging
from pathlib import Path
from datetime import datetime


def setup_logger(log_file: Path = None, log_level=logging.INFO):
    """
    로거 설정
    
    Args:
        log_file: 로그 파일 경로 (None이면 logs/pipeline_YYYYMMDD_HHMMSS.log)
        log_level: 로그 레벨
    
    Returns:
        logger 객체
    """
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"pipeline_{timestamp}.log"
    
    # 로거 생성
    logger = logging.getLogger("pipeline")
    logger.setLevel(log_level)
    
    # 기존 핸들러 제거
    logger.handlers = []
    
    # 파일 핸들러
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 콘솔 핸들러 (간단한 형식)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # 콘솔에는 경고만
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger initialized. Log file: {log_file}")
    
    return logger, log_file
