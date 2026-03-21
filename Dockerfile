# Use Python 3.13 slim image for smaller size and cross-platform compatibility
FROM python:3.13-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    # OpenCV headless mode - works without GUI
    OPENCV_VIDEOIO_PRIORITY_MSMF=0

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV core dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    libglx0 \
    libglvnd0 \
    # X11 libraries (required by OpenCV)
    libxcb1 \
    libx11-6 \
    libxau6 \
    libxdmcp6 \
    # Video processing
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    # Camera utilities
    v4l-utils \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY vision/ ./vision/

# Create output directory for potential video/image saves
RUN mkdir -p /app/output

# Create a non-root user for security (works on all platforms)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port if needed for future web interface
# EXPOSE 8080

# Set the default command
CMD ["python", "files/camera.py"]
