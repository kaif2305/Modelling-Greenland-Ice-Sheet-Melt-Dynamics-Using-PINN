FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# THIS LINE REPLACES '-e .' - It tells Python where your code lives
ENV PYTHONPATH="/app/src"

WORKDIR /app

# 1. Install system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 git && \
    rm -rf /var/lib/apt/lists/*

# 2. Pre-install heavy libraries (creates a permanent cache layer)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir dagshub mlflow joblib

# 3. Install remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your code LAST
COPY . .

CMD ["python", "main.py"]