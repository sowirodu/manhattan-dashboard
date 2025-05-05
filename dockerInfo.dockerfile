FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (if any needed for your packages)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all other files
COPY . .

# Set production mode
ENV DASH_DEBUG_MODE=false

EXPOSE 8050
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "app:server"]