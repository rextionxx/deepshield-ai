FROM python:3.11-slim

WORKDIR /app

# System libraries OpenCV needs that aren't in the slim image
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --no-deps keras==3.13.2

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
