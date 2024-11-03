# Handwriting OCR Pipeline

A powerful handwriting recognition pipeline that combines shadow removal preprocessing with state-of-the-art OCR models. Supports both local inference using MiniCPM-V and cloud-based transcription using Claude 3.

## Features

- Shadow removal preprocessing for improved image quality
- Support for both local (MiniCPM-V) and cloud (Claude 3) inference
- Batch processing capabilities
- Streaming output support
- Specialized prompting for academic content
- CLI tool for easy usage
- Python API for integration into other applications

## Installation

```bash
# Using pip
pip install git+https://github.com/sjvrensburg/handwriting-ocr.git

# Using Poetry
poetry add git+https://github.com/sjvrensburg/handwriting-ocr.git
```

## Command Line Interface

The package provides a command-line interface through the `ocr` command.

### Global Options

```bash
ocr [OPTIONS] COMMAND [ARGS]...

Options:
  --device TEXT          Device to run on (cuda/cpu) [default: cuda]
  --model TEXT          Model name/path [default: openbmb/MiniCPM-V-2_6-int4]
  --use-claude          Use Claude API instead of local model
  --no-claude           Use local model (default)
  --api-key TEXT        Anthropic API key (can also be set via ANTHROPIC_API_KEY env var)
  --help               Show this message and exit
```

### Transcribe Single Image

```bash
ocr transcribe [OPTIONS] IMAGE_PATH

Options:
  -o, --output TEXT                     Output file path
  -c, --content-type TEXT              Type of content being transcribed [default: academic notes]
  -k, --keywords TEXT                  Keywords expected in the content (can be used multiple times)
  -p, --custom-prompt TEXT             Optional custom prompt override
  --save-preprocessed / --no-save-preprocessed
                                      Save preprocessed image [default: False]
  --stream / --no-stream              Stream output tokens [default: False]
  --temperature FLOAT                 Generation temperature [default: 0.7]
  --max-tokens INTEGER               Maximum tokens to generate (Claude only) [default: 1024]
  --help                             Show this message and exit
```

### Batch Process Directory

```bash
ocr batch [OPTIONS] INPUT_DIR OUTPUT_DIR

Options:
  -c, --content-type TEXT              Type of content being transcribed [default: academic notes]
  -k, --keywords TEXT                  Keywords expected in the content (can be used multiple times)
  -p, --custom-prompt TEXT             Optional custom prompt override
  --save-preprocessed / --no-save-preprocessed
                                      Save preprocessed images [default: False]
  -e, --extensions TEXT               Comma-separated list of file extensions to process [default: .jpg,.jpeg,.png]
  --max-tokens INTEGER               Maximum tokens to generate (Claude only) [default: 1024]
  --help                             Show this message and exit
```

### Working with Prompts

The pipeline provides several ways to view and verify prompts before processing:

#### Viewing Generated Prompts
Use the `show-prompt` command to see how your parameters will be converted into a prompt:

```bash
# View default prompt for academic notes
ocr show-prompt

# View prompt for math notes with keywords
ocr show-prompt "math notes" -k "calculus" -k "derivatives"

# View prompt with custom additions
ocr show-prompt "physics notes" -k "quantum" -p "Focus on equations and diagrams"
```

#### Preview Mode
Both `transcribe` and `batch` commands support a preview mode that shows the prompt and requires confirmation before processing:

```bash
# Preview prompt before transcribing
ocr transcribe image.jpg --preview

# Preview prompt for batch processing
ocr batch ./input ./output --preview -k "chemistry" -k "reactions"
```

The preview mode helps you:
- Verify keyword integration
- Check prompt structure
- Confirm content type settings
- Review custom prompt additions
- Ensure proper context before processing

This is particularly useful when:
- Fine-tuning prompts for specific domains
- Debugging recognition issues
- Working with new content types
- Testing keyword combinations

### Streaming Output

The pipeline supports streaming output mode, which provides real-time transcription results as they're generated. The behavior differs slightly between local models and Claude API.

#### Local Model Streaming
When using local models, streaming provides token-by-token output as transcription occurs. This is useful for:
- Getting immediate feedback on transcription
- Processing long documents with real-time output
- Interactive applications requiring token-level granularity

#### Claude API Streaming
When using Claude, streaming provides chunk-based output that may contain multiple tokens or sentences. This is beneficial for:
- Getting faster initial responses
- More natural text flow in the output
- Better handling of complete thoughts and concepts

#### Using Streaming in CLI
```bash
# Stream output with local model
ocr transcribe lecture_notes.jpg --stream

# Stream with Claude API (chunk-based output)
ocr --use-claude transcribe lecture_notes.jpg --stream

# Stream with preview and keywords
ocr transcribe math_lecture.jpg \
  --stream \
  --preview \
  -c "math notes" \
  -k "calculus" \
  -k "derivatives"

# Stream and save output
ocr --use-claude transcribe lecture.jpg \
  --stream \
  --output result.txt \
  --save-metadata
```

#### Using Streaming in Python API
```python
# Stream with local model
pipeline = HandwritingTranscriptionPipeline()
for token in pipeline.process_single_image(
    "lecture_notes.jpg",
    stream=True,
    content_type="physics lecture",
    keywords=["quantum", "mechanics"]
):
    print(token, end='')  # Print each token as it arrives

# Stream with Claude API
pipeline = HandwritingTranscriptionPipeline(use_claude=True)
for chunk in pipeline.process_single_image(
    "lecture_notes.jpg",
    stream=True,
    content_type="physics lecture",
    keywords=["quantum", "mechanics"]
):
    print(chunk, end='')  # Print each chunk as it arrives
    # Process chunks in real-time
    process_chunk(chunk)  # Your custom processing function
```

#### Error Handling in Streaming
The pipeline includes robust error handling for streaming:
- Partial results are saved if streaming is interrupted
- Clear error messages for streaming issues
- Automatic recovery and result saving when possible

#### Use Cases for Streaming
Streaming output is particularly useful for:
- Real-time transcription monitoring
- Long document processing with progress feedback
- Interactive applications requiring immediate response
- Debugging and adjusting prompts
- Batch processing with progress indication
- Building responsive user interfaces

### Keywords and Content Context

The pipeline uses a context-aware system that can be fine-tuned using keywords and content types. This helps improve transcription accuracy by providing domain-specific context to the model.

#### Keyword System
- Multiple keywords can be specified using repeated `-k` flags in CLI or as a list in the API
- Keywords help the model focus on domain-specific terminology
- Keywords can include technical terms, mathematical symbols, or common phrases expected in the content
- The model uses these keywords to:
  - Better recognize domain-specific notation
  - Correctly interpret ambiguous symbols
  - Maintain consistency in technical terminology
  - Improve accuracy in formula transcription

#### Content Types
Different content types trigger specialized processing:
- `academic notes`: Optimized for general academic notation and structure
- `math notes`: Enhanced focus on mathematical symbols and equations
- `chemistry notes`: Better recognition of chemical formulas and reactions
- `physics lecture`: Improved handling of physics notation and diagrams
- `engineering drawings`: Better processing of technical diagrams and annotations

### Example Usage

```bash
# Transcribe a single image using local model
ocr transcribe image.jpg -o transcription.txt

# Transcribe using Claude with streaming output
ocr --use-claude transcribe image.jpg --stream

# Batch process a directory of images
ocr batch ./input_images ./output_transcriptions

# Process math notes with multiple keywords to improve recognition accuracy
ocr transcribe math_notes.jpg -c "math notes" -k "calculus" -k "derivatives" -k "integration" -k "partial differential"

# Process chemistry lab notes with relevant keywords
ocr transcribe chem_notes.jpg -c "chemistry notes" -k "titration" -k "pH" -k "molarity" -k "equilibrium"

# Process physics lecture notes with domain-specific terms
ocr transcribe physics.jpg -c "physics lecture" -k "quantum" -k "mechanics" -k "wave function" -k "hamiltonian"
```

## Python API

### Basic Usage

```python
from handwriting_ocr import HandwritingTranscriptionPipeline

# Initialize pipeline with local model
pipeline = HandwritingTranscriptionPipeline()

# Or use Claude
pipeline = HandwritingTranscriptionPipeline(
    use_claude=True,
    anthropic_api_key="your-api-key"
)

# Process single image with multiple contextual keywords
result = pipeline.process_single_image(
    "image.jpg",
    content_type="math notes",
    keywords=[
        "calculus",
        "derivatives",
        "integration",
        "differential equations",
        "vector fields"
    ],
    custom_prompt="Focus on mathematical equations and symbolic notation"
)

# Process chemistry lab notes with relevant context
result = pipeline.process_single_image(
    "lab_notes.jpg",
    content_type="chemistry notes",
    keywords=[
        "titration",
        "molarity",
        "pH",
        "equilibrium",
        "reaction kinetics"
    ],
    custom_prompt="Pay special attention to chemical formulas and reaction equations"
)

# Process directory
results = pipeline.process_directory(
    "input_dir",
    "output_dir",
    content_type="lecture notes",
    save_preprocessed=True
)
```

### Pipeline Class Reference

#### Constructor

```python
HandwritingTranscriptionPipeline(
    model_name: str = "openbmb/MiniCPM-V-2_6-int4",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 1,
    use_claude: bool = False,
    anthropic_api_key: Optional[str] = None
)
```

#### Methods

##### `preprocess_image`
```python
def preprocess_image(
    self,
    image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None
) -> Image.Image
```
Preprocesses an image using shadow removal techniques.

##### `transcribe_image`
```python
def transcribe_image(
    self,
    image: Image.Image,
    prompt: str = "Transcribe the handwritten text in the image.",
    stream: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 1024
) -> Union[str, Generator]
```
Transcribes text from a preprocessed image.

##### `create_targeted_prompt`
```python
def create_targeted_prompt(
    self,
    content_type: str = "academic notes",
    keywords: List[str] = None
) -> str
```
Creates a specialized prompt for specific content types.

##### `process_single_image`
```python
def process_single_image(
    self,
    image_path: Union[str, Path],
    content_type: str = "academic notes",
    keywords: List[str] = None,
    custom_prompt: Optional[str] = None,
    save_preprocessed: bool = False,
    stream: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 1024
) -> Union[str, Generator]
```
Processes a single image through the complete pipeline.

##### `process_directory`
```python
def process_directory(
    self,
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    content_type: str = "academic notes",
    keywords: List[str] = None,
    custom_prompt: Optional[str] = None,
    save_preprocessed: bool = False,
    file_extensions: List[str] = ['.jpg', '.jpeg', '.png'],
    max_tokens: int = 1024
) -> Dict[str, str]
```
Processes all images in a directory.

