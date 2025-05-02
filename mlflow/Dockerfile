FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    mlflow \
    psycopg2-binary \
    gunicorn \
    flask

EXPOSE 5000

CMD ["mlflow", "ui", "--host", "0.0.0.0"]
