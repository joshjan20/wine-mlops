# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Expose port 5000
EXPOSE 5000

# Set environment variables (optional)
ENV MLFLOW_TRACKING_URI=http://23.22.232.131:5000

# Start the FastAPI server
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "5000"]
