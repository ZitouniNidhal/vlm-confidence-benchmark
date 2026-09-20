from abc import ABC, abstractmethod
from typing import Tuple, List, Union
from PIL import Image


class BaseVLMModel(ABC):
    """
    Abstract Base Class for Vision-Language Models (VLMs) in the confidence benchmark.
    All model wrappers (e.g. SmolVLM, Qwen2VLM) should inherit from this class.
    """

    def __init__(self, model_name: str, device: str | None = None, mock: bool = False):
        self.model_name = model_name
        self.device = device
        self.mock = mock

    @abstractmethod
    def generate(
        self,
        image: Union[Image.Image, str],
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> Tuple[str, List[float]]:
        """
        Generate answer text and output token probabilities for a given input image and prompt.

        Args:
            image: PIL Image instance or path to an image file.
            prompt: Text prompt for the VLM.
            max_new_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature for generation.

        Returns:
            Tuple containing:
                - answer_text (str): Output natural language string from the model.
                - token_probs (List[float]): Sequence of generated token probabilities.
        """
        pass
