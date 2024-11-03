"""
Command-line interface for the handwriting transcription pipeline with shadow removal preprocessing.
"""
import click
from pathlib import Path
import os
import sys
from typing import Optional, Tuple, Dict
import json
from datetime import datetime
import logging

from .pipeline import HandwritingTranscriptionPipeline, create_targeted_prompt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def preview_prompt(content_type: str, keywords: Optional[Tuple[str]], 
                  custom_prompt: Optional[str]) -> str:
    """Generate and return the prompt that would be used."""
    base_prompt = create_targeted_prompt(
        content_type=content_type,
        keywords=list(keywords) if keywords else None
    )
    
    final_prompt = base_prompt
    if custom_prompt:
        final_prompt = f"{base_prompt}\n\n{custom_prompt}"
    
    return final_prompt

def save_job_metadata(
    output_path: Path,
    content_type: str,
    keywords: Tuple[str],
    custom_prompt: Optional[str],
    model_info: Dict
) -> None:
    """Save metadata about the transcription job."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "content_type": content_type,
        "keywords": list(keywords) if keywords else [],
        "custom_prompt": custom_prompt,
        "model": model_info
    }
    
    meta_path = output_path.parent / f"{output_path.stem}_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def ensure_pipeline(ctx):
    """Initialize the pipeline if it hasn't been initialized yet."""
    if 'pipeline' not in ctx.obj:
        try:
            ctx.obj['pipeline'] = HandwritingTranscriptionPipeline(
                model_name=ctx.obj['model'],
                device=ctx.obj['device'],
                use_claude=ctx.obj['use_claude'],
                anthropic_api_key=ctx.obj['api_key']
            )
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {str(e)}")
            raise click.ClickException(str(e))

@click.group()
@click.option('--device', default='cuda', help='Device to run on (cuda/cpu)')
@click.option('--model', default='openbmb/MiniCPM-V-2_6-int4', help='Model name/path')
@click.option('--use-claude/--no-claude', default=False, 
              help='Use Claude API instead of local model')
@click.option('--api-key', help='Anthropic API key (can also be set via ANTHROPIC_API_KEY env var)')
@click.option('--verbose/--no-verbose', default=False, help='Enable verbose logging')
@click.version_option(version='0.1.0')
@click.pass_context
def cli(ctx, device: str, model: str, use_claude: bool, api_key: Optional[str], 
        verbose: bool):
    """Handwriting OCR pipeline with shadow removal preprocessing.
    
    This tool provides functionality for transcribing handwritten text from images,
    with support for both local models and Claude API integration.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    ctx.ensure_object(dict)
    
    # If using Claude without explicit API key, try to get from environment
    if use_claude and not api_key:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise click.UsageError(
                "Anthropic API key must be provided via --api-key or ANTHROPIC_API_KEY environment variable"
            )
    
    # Store configuration without initializing pipeline
    ctx.obj.update({
        'device': device,
        'model': model,
        'use_claude': use_claude,
        'api_key': api_key,
        'model_info': {
            "name": model,
            "device": device,
            "use_claude": use_claude
        }
    })

@cli.command()
@click.argument('content_type', default='academic notes')
@click.option('--keywords', '-k', multiple=True, 
              help='Keywords expected in the content')
@click.option('--custom-prompt', '-p', default=None,
              help='Optional custom prompt override')
@click.option('--output', '-o', type=click.Path(), help='Save prompt to file')
def show_prompt(content_type: str, keywords: Tuple[str], 
                custom_prompt: Optional[str], output: Optional[str]):
    """Preview the prompt that would be used for transcription.
    
    Examples:
        ocr show-prompt "math notes" -k "calculus" -k "derivatives"
        ocr show-prompt "physics notes" -k "quantum" -p "Focus on equations"
        ocr show-prompt "chemistry notes" -o prompt.txt
    """
    try:
        prompt = preview_prompt(content_type, keywords, custom_prompt)
        
        click.echo("\nGenerated Prompt:")
        click.echo("================")
        click.echo(prompt)
        
        if output:
            Path(output).write_text(prompt)
            click.echo(f"\nPrompt saved to: {output}")
            
    except Exception as e:
        logger.error(f"Error generating prompt: {str(e)}")
        raise click.ClickException(str(e))


@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--content-type', '-c', default='academic notes', 
              help='Type of content being transcribed')
@click.option('--keywords', '-k', multiple=True, 
              help='Keywords expected in the content')
@click.option('--custom-prompt', '-p', default=None,
              help='Optional custom prompt override')
@click.option('--save-preprocessed/--no-save-preprocessed', default=False,
              help='Save preprocessed image')
@click.option('--stream/--no-stream', default=False, help='Stream output tokens')
@click.option('--temperature', default=0.7, type=float, help='Generation temperature')
@click.option('--max-tokens', default=1024, type=int, 
              help='Maximum tokens to generate (Claude only)')
@click.option('--preview/--no-preview', default=False,
              help='Preview prompt before processing')
@click.option('--save-metadata/--no-save-metadata', default=True,
              help='Save job metadata alongside output')
@click.pass_context
def transcribe(ctx, image_path: str, output: Optional[str], content_type: str,
               keywords: Tuple[str], custom_prompt: Optional[str],
               save_preprocessed: bool, stream: bool, temperature: float,
               max_tokens: int, preview: bool, save_metadata: bool):
    """Transcribe text from a single image."""
    try:
        # Show prompt if preview is requested
        if preview:
            prompt = preview_prompt(content_type, keywords, custom_prompt)
            click.echo("\nPrompt Preview:")
            click.echo("==============")
            click.echo(prompt)
            
            if not click.confirm('\nProceed with transcription?'):
                click.echo('Transcription cancelled.')
                sys.exit(0)
        
        output_path = Path(output) if output else None
        if output_path and output_path.exists():
            if not click.confirm(f'\nOutput file {output_path} exists. Overwrite?'):
                click.echo('Transcription cancelled.')
                sys.exit(0)
        
        # Initialize pipeline only when needed
        ensure_pipeline(ctx)
        
        # Capture full text if streaming to a file
        full_text = []
        
        result = ctx.obj['pipeline'].process_single_image(
            image_path,
            content_type=content_type,
            keywords=list(keywords) if keywords else None,
            custom_prompt=custom_prompt,
            save_preprocessed=save_preprocessed,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        if stream:
            try:
                click.echo("\nTranscription:")
                click.echo("=============")
                sys.stdout.flush()  # Ensure buffered output is displayed
                
                for chunk in result:
                    if chunk:  # Skip empty chunks
                        click.echo(chunk, nl=False)
                        sys.stdout.flush()  # Ensure each chunk is displayed immediately
                        if output_path:
                            full_text.append(chunk)
                
                click.echo("\n")  # Add final newlines
                
                if output_path:
                    output_path.write_text(''.join(full_text))
                    if save_metadata:
                        save_job_metadata(
                            output_path,
                            content_type,
                            keywords,
                            custom_prompt,
                            ctx.obj['model_info']
                        )
                    click.echo(f"Transcription saved to: {output_path}")
            except Exception as e:
                if not any(full_text):  # If we haven't received any text
                    logger.error(f"No text received during streaming: {str(e)}")
                    raise click.ClickException("No text received from the API")
                else:
                    # If we have some text, save what we got
                    logger.warning(f"Streaming ended early: {str(e)}")
                    if output_path:
                        output_path.write_text(''.join(full_text))
                        if save_metadata:
                            save_job_metadata(
                                output_path,
                                content_type,
                                keywords,
                                custom_prompt,
                                ctx.obj['model_info']
                            )
                        click.echo(f"\nPartial transcription saved to: {output_path}")
        else:
            if output_path:
                output_path.write_text(result)
                if save_metadata:
                    save_job_metadata(
                        output_path,
                        content_type,
                        keywords,
                        custom_prompt,
                        ctx.obj['model_info']
                    )
                click.echo(f"Transcription saved to: {output_path}")
            else:
                click.echo(result)
                
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('output_dir', type=click.Path(file_okay=False))
@click.option('--content-type', '-c', default='academic notes',
              help='Type of content being transcribed')
@click.option('--keywords', '-k', multiple=True,
              help='Keywords expected in the content')
@click.option('--custom-prompt', '-p', default=None,
              help='Optional custom prompt override')
@click.option('--save-preprocessed/--no-save-preprocessed', default=False,
              help='Save preprocessed images')
@click.option('--extensions', '-e', default='.jpg,.jpeg,.png',
              help='Comma-separated list of file extensions to process')
@click.option('--max-tokens', default=1024, type=int,
              help='Maximum tokens to generate (Claude only)')
@click.option('--preview/--no-preview', default=False,
              help='Preview prompt before processing')
@click.option('--save-metadata/--no-save-metadata', default=True,
              help='Save job metadata alongside outputs')
@click.option('--skip-existing/--no-skip-existing', default=True,
              help='Skip files that already have transcriptions')
@click.pass_context
def batch(ctx, input_dir: str, output_dir: str, content_type: str,
          keywords: Tuple[str], custom_prompt: Optional[str],
          save_preprocessed: bool, extensions: str, max_tokens: int,
          preview: bool, save_metadata: bool, skip_existing: bool):
    """Process all images in a directory.
    
    Examples:
        ocr batch ./input_images ./output_transcriptions
        ocr batch ./notes ./processed -k "quantum" -k "mechanics" --preview
        ocr batch ./raw ./done --save-preprocessed --extensions=".jpg,.png"
    """
    try:
        # Show prompt if preview is requested
        if preview:
            prompt = preview_prompt(content_type, keywords, custom_prompt)
            click.echo("\nPrompt Preview:")
            click.echo("==============")
            click.echo(prompt)
            
            if not click.confirm('\nProceed with batch processing?'):
                click.echo('Batch processing cancelled.')
                sys.exit(0)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_extensions = [ext.strip() for ext in extensions.split(',')]
        
        # Create a summary log file
        summary_path = output_dir / 'batch_summary.json'
        summary = {
            "timestamp": datetime.now().isoformat(),
            "content_type": content_type,
            "keywords": list(keywords) if keywords else [],
            "custom_prompt": custom_prompt,
            "model": ctx.obj['model_info'],
            "files": {}
        }
        
        # Initialize pipeline only when needed
        ensure_pipeline(ctx)
        
        results = ctx.obj['pipeline'].process_directory(
            input_dir,
            output_dir,
            content_type=content_type,
            keywords=list(keywords) if keywords else None,
            custom_prompt=custom_prompt,
            save_preprocessed=save_preprocessed,
            file_extensions=file_extensions,
            max_tokens=max_tokens
        )
        
        # Update summary with results
        for filename, result in results.items():
            status = "SUCCESS" if not result.startswith("ERROR") else "FAILED"
            error_msg = result[6:] if status == "FAILED" else None
            
            summary["files"][filename] = {
                "status": status,
                "error": error_msg
            }
        
        # Save summary
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print results
        click.echo(f"\nProcessed {len(results)} files:")
        success_count = sum(1 for r in results.values() if not r.startswith("ERROR"))
        for filename, result in results.items():
            status = "SUCCESS" if not result.startswith("ERROR") else "FAILED"
            click.echo(f"{filename}: {status}")
        
        click.echo(f"\nSummary:")
        click.echo(f"Total files: {len(results)}")
        click.echo(f"Successful: {success_count}")
        click.echo(f"Failed: {len(results) - success_count}")
        click.echo(f"Summary saved to: {summary_path}")
        
    except Exception as e:
        logger.error(f"Error during batch processing: {str(e)}")
        raise click.ClickException(str(e))

if __name__ == '__main__':
    cli(obj={})