FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . /app
WORKDIR /app

CMD ["python", "eval.py"]
