FROM python:3.12.14-slim

ENV PYTHONHASHSEED=0 \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-lock.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-lock.txt

COPY . .

CMD ["bash", "run_full_study.sh"]
