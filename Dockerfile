# Dockerfile for deploying the Medical AI System natively
# This provides a guaranteed isolation environment for TensorFlow and Streamlit

FROM python:3.10-slim

# Install system dependencies for OpenCV and other C-bindings
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement list and install pip dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Start the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
