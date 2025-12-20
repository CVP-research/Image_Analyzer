from pathlib import Path
from typing import List, Optional, Dict, Type, Any

from .api.base import BaseVideoGenerator


class VideoGeneratorFacade:
    """
    A facade for video generation, abstracting the underlying API calls
    and allowing selection of different video generation models.
    """
    _registered_generators: Dict[str, Type[BaseVideoGenerator]] = {}

    @classmethod
    def register_generator(cls, name: str, generator_class: Type[BaseVideoGenerator]):
        """Registers a video generator class with a given name."""
        if not issubclass(generator_class, BaseVideoGenerator):
            raise ValueError(f"Generator class {generator_class.__name__} must inherit from BaseVideoGenerator")
        cls._registered_generators[name] = generator_class

    def __init__(self, generator_name: str, video_dir: Path, debug_mode: Any, **kwargs):
        """
        Initializes the facade with a specific video generator.

        Args:
            generator_name: The name of the video generator to use (e.g., "veo3").
            video_dir: The directory where generated videos should be saved.
            debug_mode: An object or flag indicating if debug mode is enabled.
            **kwargs: Additional arguments to pass to the specific generator's constructor.
        """
        if generator_name not in self.__class__._registered_generators:
            raise ValueError(f"Unknown video generator: {generator_name}. Available: {list(self.__class__._registered_generators.keys())}")
        
        generator_class = self.__class__._registered_generators[generator_name]
        self._current_generator: BaseVideoGenerator = generator_class(video_dir=video_dir, debug_mode=debug_mode, **kwargs)

    def generate_video(
        self,
        selected_images: List[Path],
        additional_prompt: str = None,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generates a video using the selected video generator.
        Delegates the call to the underlying generator.
        """
        return self._current_generator.generate_video(selected_images, additional_prompt, output_path)
