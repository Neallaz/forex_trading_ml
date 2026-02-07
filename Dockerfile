# Stage 1: Base Image with system dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    wget \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib dependencies and build
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Stage 2: Dependencies installation
FROM base as dependencies

# Copy requirements file
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Development stage (for model training)
FROM dependencies as development

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/raw \
    /app/data/processed \
    /app/models/ml \
    /app/models/dl \
    /app/models/ensemble \
    /app/trading/backtesting/results \
    /app/dashboard/static

# Set permissions
RUN chmod +x /app/main.py \
    && chmod +x /app/data/scripts/*.py \
    && chmod +x /app/models/*/*.py \
    && chmod +x /app/trading/*/*.py \
    && chmod +x /app/trading/backtesting/backtester.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health', timeout=2)" || exit 1

# Expose ports
EXPOSE 8501  # Streamlit
EXPOSE 8000  # FastAPI (if added later)

# Default command (run dashboard)
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Stage 4: Production stage (lightweight)
FROM python:3.9-slim as production

# Copy only necessary files from development stage
COPY --from=development /usr/lib /usr/lib
COPY --from=development /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=development /app /app

# Set working directory
WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 forexuser && \
    chown -R forexuser:forexuser /app

USER forexuser

# Expose ports
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health', timeout=2)" || exit 1

# Run dashboard
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]