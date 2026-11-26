# MP074 prediction API image
FROM python:3.12-slim

WORKDIR /app

# Install pinned libraries first, so this layer is cached when only code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code and the committed champion model come after the install step.
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model.joblib

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
