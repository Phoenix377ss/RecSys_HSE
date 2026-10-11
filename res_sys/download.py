from pathlib import Path

from huggingface_hub import snapshot_download
from loguru import logger
import typer

from res_sys.config import RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    repo_id: str = "t-tech/T-ECD",
    allow_patterns: str = "dataset/small/*",
    output_dir: Path = RAW_DATA_DIR,
) -> None:

    logger.info(f"Downloading files matching '{allow_patterns}' from HF repo '{repo_id}'...")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=allow_patterns,
            local_dir=output_dir,
        )
        logger.success(f"T-ECD dataset files successfully downloaded into: {output_dir}")

    except Exception as e:
        logger.error(f"Failed to download dataset via snapshot_download: {e}")
        raise typer.Exit(code=1) from None
    # -----------------------------------------


if __name__ == "__main__":
    app()
