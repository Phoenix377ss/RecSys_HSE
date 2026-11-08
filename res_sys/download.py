import os
from pathlib import Path

from res_sys.config import RAW_DATA_DIR

# Кэш HF держим внутри проекта (data/.hf_cache): $HOME/.cache не везде доступен
# для записи, а кэш на несколько ГБ не должен засорять домашнюю папку.
os.environ.setdefault("HF_HOME", str(RAW_DATA_DIR.parent / ".hf_cache"))

from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402
from loguru import logger  # noqa: E402
import typer  # noqa: E402

app = typer.Typer()


@app.command()
def main(
    repo_id: str = "t-tech/T-ECD",
    allow_patterns: str = "dataset/small/marketplace/*",
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

        hf_hub_download(
            repo_id="t-tech/T-ECD",
            repo_type="dataset",
            filename="dataset/full/users.pq",
            local_dir=RAW_DATA_DIR
)

    except Exception as e:
        logger.error(f"Failed to download dataset via snapshot_download: {e}")
        raise typer.Exit(code=1) from None
    # -----------------------------------------


if __name__ == "__main__":
    app()
