# Lightweight inference image: ONNX Runtime only, no PyTorch/training stack.
# The serving import chain depends solely on numpy, scipy, onnxruntime and
# FastAPI, so the heavy training dependencies are never installed here.
#
# Build (export a bundle first; see README "Serving & Deployment"):
#   docker build -t deep-ecg-serve .
# Run:
#   docker run -p 8000:8000 deep-ecg-serve
# Serve a different bundle without rebuilding (mount it and point MODEL_DIR at it):
#   docker run -p 8000:8000 -v "$PWD/artifacts:/app/artifacts" \
#     -e MODEL_DIR=/app/artifacts/ensemble deep-ecg-serve

FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    MODEL_DIR=/app/artifacts/resnet1d \
    BACKEND=onnx

RUN pip install --no-cache-dir \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.30" \
    "onnxruntime>=1.18" \
    "numpy>=1.26" \
    "scipy>=1.13" \
    "pydantic>=2.7"

COPY src/ /app/src/
COPY artifacts/ /app/artifacts/

EXPOSE 8000
CMD ["uvicorn", "deep_ecg.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
