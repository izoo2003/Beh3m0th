FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY evals ./evals

RUN pip install --no-cache-dir ".[ui]"

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "src/beh3m0th/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
