FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && python -c "import numpy, torch; print('numpy', numpy.__version__); print('torch', torch.__version__); torch.tensor([1.0]).numpy()"

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "gunicorn app:app --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:${PORT}"]
