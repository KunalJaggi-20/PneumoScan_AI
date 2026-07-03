FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for image processing
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install core Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install CPU-optimized light-weight PyTorch (avoids downloading huge 1.5GB GPU weights)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy app code
COPY . .

# Expose server port
EXPOSE 5000

# Start server using Gunicorn WSGI
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "server:app"]
