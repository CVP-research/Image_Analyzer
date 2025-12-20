import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import google.genai as genai
from google.genai import types

from .base import BaseVideoGenerator
from ..prompt import PromptBuilder

class Veo3VideoGenerator(BaseVideoGenerator):
    def __init__(
        self,
        video_dir: Path,
        debug_mode,
        credentials_path: Optional[Path] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        max_reference_images: int = 3,
        generation_timeout: int = 300  # seconds
    ):
        super().__init__(video_dir, debug_mode)
        self.client = None
        self.max_reference_images = max_reference_images
        self.generation_timeout = generation_timeout

        if not self.debug_mode.enabled:
            try:
                # 인증 키 경로 설정
                if credentials_path:
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)
                elif "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
                    raise ValueError("Google credentials path not provided.")

                self.client = genai.Client(
                    vertexai=True,
                    project=project or "default-project",
                    location=location or "us-central1"
                )
            except Exception as e:
                print(f"Error initializing Google Generative AI client: {e}")
                print("(디버그 모드로 실행하려면: python pipeline/run.py --debug)")
                sys.exit(1)
        else:
            print("DEBUG MODE: Google Generative AI client 초기화 스킵됨")

    def generate_video(
        self,
        selected_images: List[Path],
        additional_prompt: str = None,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Veo3 API 호출하여 360도 영상 생성
        Debug 모드이면 실제 API 호출 대신 더미 파일 생성
        """
        if self.debug_mode.enabled:
            print("[DEBUG MODE] Skipping actual video generation.")
            dummy_path = output_path or self.video_dir / f"dummy_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            dummy_path.touch()
            return dummy_path
        
        if len(selected_images) > self.max_reference_images:
            print(f"Warning: 선택된 이미지가 {self.max_reference_images}장을 초과합니다. 처음 {self.max_reference_images}장만 사용됩니다.")

        print("\n[Step] Generating 360° video with Veo3 API...")
        prompt_text = PromptBuilder.build_prompt(additional_prompt)

        reference_images = [
            types.VideoGenerationReferenceImage(
                image=types.Image.from_file(location=str(file)),
                reference_type="asset",
            )
            for file in selected_images[:self.max_reference_images]
        ]

        try:
            operation = self.client.models.generate_videos(
                model="veo-3.1-generate-preview",
                source=types.GenerateVideosSource(prompt=prompt_text),
                config=types.GenerateVideosConfig(
                    reference_images=reference_images,
                    aspect_ratio="16:9",
                ),
            )

            start_time = time.time()
            while not operation.done:
                if time.time() - start_time > self.generation_timeout:
                    raise TimeoutError("Video generation exceeded the timeout limit.")
                print("Waiting for video generation...")
                time.sleep(15)
                # operation.name으로 재조회
                operation = self.client.operations.get(operation.name)

            print("✓ Video generation completed!")

            if operation.response and operation.result.generated_videos:
                video_bytes = operation.result.generated_videos[0].video.video_bytes
                local_path = output_path or self.video_dir / f"generated_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                with open(local_path, "wb") as f:
                    f.write(video_bytes)
                print(f"✓ Saved generated video locally: {local_path}")
                return local_path
            else:
                raise RuntimeError("Video generation failed: No videos returned.")

        except Exception as e:
            print(f"Veo API Exception: {e}")
            raise e
