from PIL import Image
from models.base import BaseVLMModel
from models.smolvlm import SmolVLM
from models.qwen2vl import Qwen2VLM


def test_smolvlm_mock_generation():
    model = SmolVLM(mock=True)
    assert isinstance(model, BaseVLMModel)
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    prompt = "Please answer with confidence."
    answer, probs = model.generate(img, prompt)

    assert isinstance(answer, str)
    assert len(answer) > 0
    assert isinstance(probs, list)
    assert len(probs) > 0
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_qwen2vl_mock_generation():
    model = Qwen2VLM(mock=True)
    assert isinstance(model, BaseVLMModel)
    img = Image.new("RGB", (64, 64), color=(0, 255, 0))
    prompt = "Please answer with confidence."
    answer, probs = model.generate(img, prompt)

    assert isinstance(answer, str)
    assert len(answer) > 0
    assert isinstance(probs, list)
    assert len(probs) > 0
    assert all(0.0 <= p <= 1.0 for p in probs)
