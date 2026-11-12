FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend /app
EXPOSE 8000
CMD ["python", "-m", "app.web_main"]
