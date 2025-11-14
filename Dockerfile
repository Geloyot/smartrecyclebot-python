# smartrecyclebot-python/Dockerfile
FROM python:3.11-slim

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . /app

# Create models directory if it doesn't exist
RUN mkdir -p /app/models

ENV PYTHONUNBUFFERED=1

# Expose the port (Render uses PORT env variable, but document 8000 as default)
EXPOSE 8000

# For production with Gunicorn (recommended)
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-"]