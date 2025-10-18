# Use a lightweight Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set work directory
WORKDIR /app

# System deps (optional, kept minimal for wheels)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app

# Create logs directory (match app expectations)
RUN mkdir -p /app/logs

# Copy and prepare startup script to run both FastAPI and Streamlit
RUN chmod +x /app/start.sh

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501

# Start both services
CMD ["./start.sh"]
