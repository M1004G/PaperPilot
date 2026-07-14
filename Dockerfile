# Single image, used for both the backend (FastAPI) and frontend (Streamlit)
# services -- docker-compose.yml runs the same image with different commands.
# Kept as one image rather than two to avoid installing the (large) shared
# dependency set -- sentence-transformers/torch/faiss -- twice.
FROM python:3.11-slim

WORKDIR /app

# System deps: PyMuPDF and faiss-cpu need a C++ runtime; curl is used by the
# HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY frontend/ frontend/
COPY scripts/ scripts/

# data/ is where SQLite persistence lives -- mounted as a volume in
# docker-compose.yml so it survives container recreation, not just restarts.
RUN mkdir -p data

EXPOSE 8000 8501

# Default command; overridden per-service in docker-compose.yml.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
