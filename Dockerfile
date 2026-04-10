# ---- Base image ----
FROM python:3.11-slim

# Hugging Face Spaces requirement: run as non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY env/      ./env/
COPY app.py    .
COPY inference.py .
COPY openenv.yaml .

# Ensure Python can find the env package
ENV PYTHONPATH=/app

# Switch to non-root user
USER appuser

# Expose port (HF Spaces uses 7860 by default)
EXPOSE 7860

# Health-check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Entrypoint
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
