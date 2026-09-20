import pytest
from PIL import Image
from experiments.run_single_image import run_single_image_eval


def test_run_single_image_eval_mock(tmp_path):
    img_path = tmp_path / "test_img.jpg"
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    img.save(img_path)

    plot_path = tmp_path / "output_plot.png"

    res = run_single_image_eval(
        image_path=str(img_path),
        model_type="smolvlm",
        mock=True,
        degradation="blur",
        severity="mid",
        ground_truth="pizza",
        output_plot=str(plot_path),
    )

    assert "answer_text" in res
    assert "verbalized_confidence" in res
    assert "internal_confidence" in res
    assert "sequence_entropy" in res
    assert plot_path.exists()
