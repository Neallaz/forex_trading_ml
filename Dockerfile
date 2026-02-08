FROM python:3.12-slim

WORKDIR /app

# 1. فقط ضروری‌ترین dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. اول numpy و scipy با wheel
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.13.0

# 3. کپی requirements
COPY requirements.txt .

# 4. نصب بقیه
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py", "--mode", "train"]

# # Dockerfile - با رفع کامل مشکلات
# FROM python:3.12-slim

# WORKDIR /app

# # 1. ابتدا system dependencies را نصب کن
# RUN apt-get update && apt-get install -y \
#     gcc \
#     g++ \
#     make \
#     wget \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # 2. آپدیت pip و setuptools
# RUN pip install --upgrade pip setuptools wheel

# # 3. فایل requirements را کپی کن
# COPY requirements.txt .

# # 4. **مهم: ابتدا numpy و scipy را جداگانه نصب کن**
# RUN pip install \
#     numpy==1.26.4 \
#     scipy==1.13.0

# # 5. سپس بقیه requirements را نصب کن
# RUN pip install -r requirements.txt

# # 6. کدهای پروژه
# COPY . .

# # 7. ایجاد پوشه‌ها
# RUN mkdir -p /app/data /app/models /app/logs

# EXPOSE 8501

# CMD ["python", "main.py", "--mode", "train"]

# # # Dockerfile - Python 3.12 Optimized
# # FROM python:3.12-slim

# # # تنظیمات
# # ENV PYTHONUNBUFFERED=1 \
# #     PYTHONPATH=/app \
# #     DEBIAN_FRONTEND=noninteractive \
# #     PIP_DEFAULT_TIMEOUT=100 \
# #     PIP_NO_CACHE_DIR=off \
# #     PIP_DISABLE_PIP_VERSION_CHECK=on

# # WORKDIR /app

# # # 1. System Dependencies
# # RUN apt-get update && apt-get install -y \
# #     gcc \
# #     g++ \
# #     make \
# #     wget \
# #     curl \
# #     git \
# #     build-essential \
# #     libpq-dev \
# #     gfortran \          
# #     libopenblas-dev \   
# #     liblapack-dev \   
# #     pkg-config \  
# #     && rm -rf /var/lib/apt/lists/*

# # # 2. Upgrade pip first
# # RUN pip install  --upgrade \
# #     pip==24.0 \
# #     wheel==0.43.0 \
# #     setuptools==70.0.0

# # # 3. Install CORE packages first (no dependencies conflict)
# # RUN pip install \
# #     numpy==1.26.4 \
# #     pandas==2.2.1 \
# #     scipy==1.13.0 \
# #     scikit-learn==1.5.0 \
# #     cython==3.0.10

# # # 4. Install TensorFlow CPU (lighter, no GPU dependencies)
# # RUN pip install  \
# #     tensorflow-cpu==2.16.2 \
# #     keras==3.3.0

# # # 5. Copy requirements and install the rest
# # COPY requirements.txt .
# # # RUN pip install --no-cache-dir -r requirements.txt
# # RUN pip install -r requirements.txt


# # # 6. Copy project code
# # COPY . .

# # # 7. Create directories
# # RUN mkdir -p \
# #     /app/data/raw \
# #     /app/data/processed \
# #     /app/models/ml \
# #     /app/models/dl \
# #     /app/logs \
# #     /app/results \
# #     /app/notebooks

# # # 8. Set permissions
# # RUN chmod -R 755 /app

# # # 9. Expose ports
# # EXPOSE 8501 8000 8888

# # # 10. Health check
# # HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
# #     CMD python -c "import sys; sys.exit(0)"

# # # 11. Entry point
# # CMD ["python", "main.py", "--mode", "train"]




# # # Dockerfile - Python 3.12 Optimized
# # FROM python:3.12-slim

# # # تنظیمات
# # ENV PYTHONUNBUFFERED=1 \
# #     PYTHONPATH=/app \
# #     DEBIAN_FRONTEND=noninteractive \
# #     PIP_DEFAULT_TIMEOUT=100 \
# #     PIP_NO_CACHE_DIR=off \
# #     PIP_DISABLE_PIP_VERSION_CHECK=on

# # WORKDIR /app

# # # 1. System Dependencies
# # RUN apt-get update && apt-get install -y \
# #     gcc \
# #     g++ \
# #     make \
# #     wget \
# #     curl \
# #     git \
# #     build-essential \
# #     libpq-dev \
# #     gfortran \          
# #     libopenblas-dev \   
# #     liblapack-dev \   
# #     pkg-config \  
# #     && rm -rf /var/lib/apt/lists/*

# # # 2. Upgrade pip first
# # RUN pip install --no-cache-dir --upgrade \
# #     pip==24.0 \
# #     wheel==0.43.0 \
# #     setuptools==70.0.0

# # # 3. Install CORE packages first (no dependencies conflict)
# # RUN pip install --no-cache-dir \
# #     numpy==1.26.4 \
# #     pandas==2.2.1 \
# #     scipy==1.13.0 \
# #     scikit-learn==1.5.0 \
# #     cython==3.0.10

# # # 4. Install TensorFlow CPU (lighter, no GPU dependencies)
# # RUN pip install --no-cache-dir \
# #     tensorflow-cpu==2.16.2 \
# #     keras==3.3.0

# # # 5. Copy requirements and install the rest
# # COPY requirements.txt .
# # RUN pip install --no-cache-dir -r requirements.txt

# # # 6. Copy project code
# # COPY . .

# # # 7. Create directories
# # RUN mkdir -p \
# #     /app/data/raw \
# #     /app/data/processed \
# #     /app/models/ml \
# #     /app/models/dl \
# #     /app/logs \
# #     /app/results \
# #     /app/notebooks

# # # 8. Set permissions
# # RUN chmod -R 755 /app

# # # 9. Expose ports
# # EXPOSE 8501 8000 8888

# # # 10. Health check
# # HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
# #     CMD python -c "import sys; sys.exit(0)"

# # # 11. Entry point
# # CMD ["python", "main.py", "--mode", "train"]