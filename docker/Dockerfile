FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY . .

CMD ["python", "app.py"]
