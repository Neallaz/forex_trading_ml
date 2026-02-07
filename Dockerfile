# Dockerfile - Python 3.12
FROM python:3.12-slim

# تنظیمات سیستم
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1. نصب system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    wget \
    curl \
    git \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. نصب TA-Lib (اختیاری - می‌تونید کامنت کنید اگر مشکل داشت)
# RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
#     && tar -xzf ta-lib-0.4.0-src.tar.gz \
#     && cd ta-lib/ \
#     && ./configure --prefix=/usr \
#     && make \
#     && make install \
#     && cd .. \
#     && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# 3. آپدیت pip
RUN pip install --no-cache-dir --upgrade pip==24.0 wheel==0.42.0 setuptools==70.0.0

# 4. ابتدا پکیج‌های اصلی را نصب کن
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.0 \
    scipy==1.13.0 \
    scikit-learn==1.4.1.post1 \
    cython==3.0.10

# 5. کپی requirements
COPY requirements.txt .

# 6. نصب بقیه پکیج‌ها
RUN pip install --no-cache-dir -r requirements.txt

# 7. کپی کدهای پروژه
COPY . .

# 8. ایجاد پوشه‌ها
RUN mkdir -p /app/data/raw /app/data/processed /app/models /app/logs /app/results

# 9. پورت‌ها
EXPOSE 8501 8000 8888

# 10. نقطه ورود
CMD ["python", "main.py", "--mode", "train"]