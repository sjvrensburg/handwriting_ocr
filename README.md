# Handwriting OCR

A Python library and CLI tool for OCR of handwritten text using MiniCPM-V with shadow removal preprocessing.

## Installation

```bash
git clone https://github.com/sjvrensburg/handwriting_ocr.git
cd handwriting_ocr
poetry install
```

## Library Usage

```python
from handwriting_ocr import HandwritingTranscriptionPipeline

pipeline = HandwritingTranscriptionPipeline(
    model_name="openbmb/MiniCPM-V-2_6-int4",
    device="cuda"  # or "cpu"
)

# Single image
result = pipeline.process_single_image(
    "image.jpg",
    content_type="academic notes",
    keywords=["calculus", "integration"]
)

# Batch processing
results = pipeline.process_directory(
    "input_directory",
    "output_directory"
)

# Custom prompting
prompt = pipeline.create_targeted_prompt(
    content_type="lecture notes",
    keywords=["quantum mechanics"]
)
result = pipeline.process_single_image("image.jpg", custom_prompt=prompt)

# Streaming output
for chunk in pipeline.process_single_image("image.jpg", stream=True):
    print(chunk, end="", flush=True)
```

## CLI Usage

```bash
# Single image
ocr single input.jpg -o output.txt \
    --content-type "math homework" \
    --keywords "calculus" "derivatives" \
    --save-preprocessed

# Batch processing
ocr batch input_dir/ output_dir/ \
    --save-preprocessed \
    --extensions ".jpg,.png"
```

### Options

#### Global
- `--device`: cuda/cpu (default: cuda)
- `--model`: Model name/path

#### Single Image
- `-o, --output`: Output file path
- `-t, --content-type`: Content type
- `-k, --keywords`: Keywords (multiple)
- `-p, --custom-prompt`: Custom prompt
- `--save-preprocessed`: Save preprocessed image
- `--stream`: Stream output
- `--temperature`: Generation temperature

#### Batch
- `-p, --prompt`: Custom prompt
- `--save-preprocessed`: Save preprocessed images
- `-e, --extensions`: File extensions

## Environment Variables
```bash
export OCR_MODEL="alternate/model/path"
export OCR_DEVICE="cpu"
export OCR_BATCH_SIZE="2"
```

## Testing
```bash
poetry run pytest
```
