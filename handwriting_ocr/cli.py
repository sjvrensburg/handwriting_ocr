import click
from pathlib import Path
import os
from typing import Optional, Tuple
from .pipeline import HandwritingTranscriptionPipeline

@click.group()
@click.option('--device', default='cuda', help='Device to run on (cuda/cpu)')
@click.option('--model', default='openbmb/MiniCPM-V-2_6-int4', help='Model name/path')
@click.option('--use-claude/--no-claude', default=False, 
              help='Use Claude API instead of local model')
@click.option('--api-key', help='Anthropic API key (can also be set via ANTHROPIC_API_KEY env var)')
@click.pass_context
def cli(ctx, device: str, model: str, use_claude: bool, api_key: Optional[str]):
    """CLI tool for handwriting transcription with shadow removal preprocessing."""
    ctx.ensure_object(dict)
    
    # If using Claude without explicit API key, try to get from environment
    if use_claude and not api_key:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise click.UsageError(
                "Anthropic API key must be provided via --api-key or ANTHROPIC_API_KEY environment variable"
            )
    
    ctx.obj['pipeline'] = HandwritingTranscriptionPipeline(
        model_name=model,
        device=device,
        use_claude=use_claude,
        anthropic_api_key=api_key
    )

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
@click.pass_context
def transcribe(ctx, image_path: str, output: Optional[str], content_type: str,
               keywords: Tuple[str], custom_prompt: Optional[str],
               save_preprocessed: bool, stream: bool, temperature: float,
               max_tokens: int):
    """Transcribe text from a single image."""
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
        for chunk in result:
            click.echo(chunk, nl=False)
    else:
        if output:
            Path(output).write_text(result)
        else:
            click.echo(result)

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
@click.pass_context
def batch(ctx, input_dir: str, output_dir: str, content_type: str,
          keywords: Tuple[str], custom_prompt: Optional[str],
          save_preprocessed: bool, extensions: str, max_tokens: int):
    """Process all images in a directory."""
    file_extensions = [ext.strip() for ext in extensions.split(',')]
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
    
    # Print summary
    click.echo(f"\nProcessed {len(results)} files:")
    for filename, result in results.items():
        status = "SUCCESS" if not result.startswith("ERROR") else "FAILED"
        click.echo(f"{filename}: {status}")

if __name__ == '__main__':
    cli(obj={})