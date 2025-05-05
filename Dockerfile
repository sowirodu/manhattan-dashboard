# syntax=docker/dockerfile:1

FROM python:3.13-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dataset and requirements file first for caching
COPY dataset_to_model_with.csv .
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variable for production
ENV DASH_DEBUG_MODE=false

# Expose the port used by the Dash/Gunicorn app
EXPOSE 8050

# Command to run the app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "app:server"]
