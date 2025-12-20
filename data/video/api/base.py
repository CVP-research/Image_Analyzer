import abc
from pathlib import Path
from typing import List, Optional, Any


class BaseVideoGenerator(abc.ABC):
    """
    Abstract base class for all video generation models.
    Defines the common interface for different video generation APIs.
    """

    def __init__(self, video_dir: Path, debug_mode: Any):
        """
        Initializes the base video generator.

        Args:
            video_dir: The directory where generated videos should be saved.
            debug_mode: An object or flag indicating if debug mode is enabled.
                        This is passed to allow concrete implementations to
                        handle debug-specific logic (e.g., skipping API calls).
        """
        self.video_dir = video_dir
        self.debug_mode = debug_mode

    @abc.abstractmethod
    def generate_video(
        self,
        selected_images: List[Path],
        additional_prompt: str = None,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generates a video based on reference images and a prompt.

        Args:
            selected_images: A list of Paths to reference images for video generation.
            prompt: The text prompt to guide the video generation.
            output_path: Optional; the specific path to save the generated video.
                         If None, the implementation should decide a default path
                         within self.video_dir.

        Returns:
            The Path to the generated video file.
        """
        raise NotImplementedError
