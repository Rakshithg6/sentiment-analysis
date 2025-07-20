FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./

# Install git (needed for some pip installs), upgrade pip, set pip timeout
RUN apt-get update && apt-get install -y git \
    && pip install --upgrade pip \
    && pip config set global.timeout 300 \
    && pip install --no-cache-dir -r requirements.txt --timeout 300

COPY app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
