FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for caching and output
RUN mkdir -p /app/.cache /app/generated_textures

# Set environment variables
ENV MODEL_CACHE_DIR=/app/.cache
ENV OUTPUT_DIR=/app/generated_textures
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run the application
CMD ["gunicorn", "web_app:app", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "300"]
