import tempfile
from pathlib import Path
from PIL import Image
from data.prepare_custom_dataset import create_jsonl_from_local_folder


def test_create_jsonl_from_local_folder():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cat_dir = tmp_path / "pizza"
        cat_dir.mkdir(parents=True, exist_ok=True)

        img_path = cat_dir / "sample_1.jpg"
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(img_path)

        out_jsonl = tmp_path / "output.jsonl"
        res_file = create_jsonl_from_local_folder(
            image_dir=str(tmp_path),
            output_jsonl=str(out_jsonl),
        )

        assert Path(res_file).exists()
        content = out_jsonl.read_text(encoding="utf-8")
        assert "pizza" in content
        assert "sample_1.jpg" in content
