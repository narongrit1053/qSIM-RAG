FROM python:3.10-slim

WORKDIR /app

# Install build dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip

# Install standard requirements with increased timeout for slow connections
RUN pip install --no-cache-dir -r requirements.txt --default-timeout=1000

# Install llama-cpp-python separately to ensure pre-built wheel usage and isolate failure
RUN pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    --no-cache-dir \
    --default-timeout=1000

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
