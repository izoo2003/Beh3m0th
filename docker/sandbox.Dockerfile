FROM python:3.12-slim

RUN pip install --no-cache-dir pytest

WORKDIR /workspace

CMD ["python", "-c", "print('beh3m0th-sandbox')"]
