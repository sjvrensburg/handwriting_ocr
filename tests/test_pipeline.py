import pytest
from pathlib import Path
from PIL import Image
from handwriting_ocr import HandwritingTranscriptionPipeline

def test_pipeline_initialization():
    pipeline = HandwritingTranscriptionPipeline(device="cpu")
    assert pipeline is not None
    assert pipeline.device == "cpu"

def test_image_preprocessing(tmp_path):
    pipeline = HandwritingTranscriptionPipeline(device="cpu")
    test_image = Image.new('RGB', (100, 100), color='white')
    image_path = tmp_path / "test.jpg"
    test_image.save(image_path)
    processed_image = pipeline.preprocess_image(image_path)
    assert isinstance(processed_image, Image.Image)