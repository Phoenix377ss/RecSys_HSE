FROM ghcr.io/mlflow/mlflow:v3.18.0

# uv образ
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN uv pip install --system --no-cache psycopg2-binary boto3

WORKDIR /mlflow
EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--artifacts-destination", "./mlruns", \
     "--serve-artifacts" , \
     "--allowed-hosts", "localhost,localhost:5000,127.0.0.1,127.0.0.1:5000,mlflow,mlflow:5000"]