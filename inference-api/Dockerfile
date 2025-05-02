FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . /app
WORKDIR /app

# Expose Flask port
EXPOSE 5002

CMD ["python", "app.py"]