from pathlib import Path
import os
from typing import Optional, Union, List, Dict, Generator
import base64

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from shadow_removal import ShadowRemovalPipeline
import anthropic
from anthropic import Anthropic

class HandwritingTranscriptionPipeline:
    """Pipeline for transcribing handwritten text from images with shadow removal preprocessing."""
    
    def __init__(
        self,
        model_name: str = "openbmb/MiniCPM-V-2_6-int4",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 1,
        use_claude: bool = False,
        anthropic_api_key: Optional[str] = None
    ):
        self.device = device
        self.batch_size = batch_size
        self.use_claude = use_claude
        
        # Initialize shadow removal pipeline
        self.shadow_removal = ShadowRemovalPipeline(device=device)
        
        if use_claude:
            if not anthropic_api_key:
                anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
                if not anthropic_api_key:
                    raise ValueError("Anthropic API key must be provided when using Claude")
            self.client = Anthropic(api_key=anthropic_api_key)
            self.model_name = "claude-3-sonnet-20240229"
        else:
            # Initialize MiniCPM-V model and tokenizer
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model.eval()
            self.model_name = model_name
            
            if device == "cuda":
                self.model.to(device)
    
    def preprocess_image(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> Image.Image:
        """Preprocess an image using shadow removal."""
        if output_path is None:
            output_path = str(Path(image_path).with_suffix('.processed.jpg'))
            
        # Process image with shadow removal
        self.shadow_removal.process_image(str(image_path), str(output_path))
        
        # Load and return the processed image
        return Image.open(output_path).convert('RGB')
    
    def _encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string for Claude API."""
        import io
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def transcribe_image(
        self,
        image: Image.Image,
        prompt: str = "Transcribe the handwritten text in the image.",
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Union[str, Generator]:
        """Transcribe text from a preprocessed image."""
        if self.use_claude:
            img_b64 = self._encode_image(image)
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }],
                stream=stream
            )
            
            if stream:
                return (chunk.delta.text for chunk in message if chunk.delta.text)
            else:
                return message.content[0].text
        else:
            msgs = [{'role': 'user', 'content': [image, prompt]}]
            
            if stream:
                return self.model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    sampling=True,
                    temperature=temperature,
                    stream=True
                )
            else:
                return self.model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=self.tokenizer
                )
    
    def create_targeted_prompt(
        self,
        content_type: str = "academic notes",
        keywords: List[str] = None
    ) -> str:
        """Create a targeted prompt for specific content types."""
        if keywords is None:
            keywords = []
            
        template = f"""Transcribe this {content_type}. The content contains these keywords: {', '.join(keywords)}.

Focus areas:
1. Mathematical notation and formulas
2. Technical terminology and definitions
3. Structural elements (lists, hierarchies)
4. Variable relationships
5. Annotations and corrections

Format equations using standard LaTeX notation where applicable."""
        return template

    def process_single_image(
        self,
        image_path: Union[str, Path],
        content_type: str = "academic notes",
        keywords: List[str] = None,
        custom_prompt: Optional[str] = None,
        save_preprocessed: bool = False,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 1024  # Added max_tokens parameter
    ) -> Union[str, Generator]:
        """Process a single image through the complete pipeline."""
        prompt = self.create_targeted_prompt(content_type=content_type, keywords=keywords)
        
        if custom_prompt:
            prompt = f"{prompt}\n\n{custom_prompt}"
        
        # Preprocess the image
        preprocessed_image = self.preprocess_image(
            image_path,
            output_path=str(Path(image_path).with_suffix('.processed.jpg')) if save_preprocessed else None
        )
        
        # Transcribe the preprocessed image
        return self.transcribe_image(
            preprocessed_image,
            prompt=prompt,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens  # Pass max_tokens to transcribe_image
        )
    
    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        content_type: str = "academic notes",
        keywords: List[str] = None,
        custom_prompt: Optional[str] = None,
        save_preprocessed: bool = False,
        file_extensions: List[str] = ['.jpg', '.jpeg', '.png'],
        max_tokens: int = 1024  # Added max_tokens parameter
    ) -> Dict[str, str]:
        """Process all images in a directory."""
        input_dir = Path(input_dir)
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for file_path in input_dir.iterdir():
            if file_path.suffix.lower() in file_extensions:
                try:
                    transcription = self.process_single_image(
                        file_path,
                        content_type=content_type,
                        keywords=keywords,
                        custom_prompt=custom_prompt,
                        save_preprocessed=save_preprocessed,
                        stream=False,
                        max_tokens=max_tokens  # Pass max_tokens parameter
                    )
                    
                    results[file_path.name] = transcription
                    
                    if output_dir:
                        output_file = output_dir / f"{file_path.stem}_transcription.txt"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(transcription)
                            
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    results[file_path.name] = f"ERROR: {str(e)}"
        
        return results