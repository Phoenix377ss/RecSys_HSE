import os
import time
import urllib.request
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def wait_for_mlflow(uri: str, timeout: int = 30) -> None:
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(f"{uri}/health") as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"MLflow не ответил за {timeout} сек.")


def test_mlflow_full_cycle(tmp_path):
    wait_for_mlflow(TRACKING_URI)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("smoke_test_experiment")

    expected_text = "smoke test payload 123"
    local_file = tmp_path / "payload.txt"
    local_file.write_text(expected_text, encoding="utf-8")

    with mlflow.start_run(run_name="automated_smoke_test") as run:
        mlflow.log_param("env", "docker_test")
        mlflow.log_metric("test_score", 1.0)
        mlflow.log_artifact(str(local_file))
        run_id = run.info.run_id

    download_dir = tmp_path / "downloaded"
    download_dir.mkdir()
    downloaded_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="payload.txt",
        dst_path=str(download_dir),
    )

    with open(downloaded_path, "r", encoding="utf-8") as f:
        assert f.read() == expected_text

    # Помечаем тестовый прогон как удаленный, чтобы не засорять UI
    MlflowClient().delete_run(run_id)
