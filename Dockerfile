# Use official Python slim image — smaller than full Python image
# WHY slim: removes docs, tests, dev tools. Production images should be minimal.
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first — WHY: Docker caches layers.
# If you copy all files first, every code change invalidates the cache.
# Copying requirements separately means pip install is cached unless
# requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the code
COPY . .

# Expose port 8000 — what the container listens on
EXPOSE 8000

# Start FastAPI with uvicorn
# --host 0.0.0.0: listen on all interfaces (required in containers)
# --workers 2: handle 2 concurrent requests
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]