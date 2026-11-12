FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend /app

EXPOSE 8000

CMD ["python", "-m", "app.web_main"]