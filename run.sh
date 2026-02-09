#!/bin/bash
# run.sh - اسکریپت اجرای آسان پروژه

set -e  # در صورت خطا اجرا متوقف شود

echo "🚀 Starting Forex ML Trading System..."

# بررسی وجود Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# بررسی وجود docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install docker-compose."
    exit 1
fi

# منو
case "$1" in
    start)
        echo "📦 Building and starting all services..."
        docker-compose up -d --build
        echo "✅ Services started. Dashboard available at: http://localhost:8501"
        ;;
    
    stop)
        echo "🛑 Stopping all services..."
        docker-compose down
        echo "✅ Services stopped."
        ;;
    
    restart)
        echo "🔄 Restarting services..."
        docker-compose restart
        echo "✅ Services restarted."
        ;;
    
    logs)
        echo "📋 Showing logs..."
        docker-compose logs -f
        ;;
    
    data)
        echo "📥 Running data pipeline..."
        docker-compose exec ml-trader python data/scripts/01_download_data.py
        docker-compose exec ml-trader python data/scripts/02_preprocess.py
        docker-compose exec ml-trader python data/scripts/03_feature_engineering.py
        echo "✅ Data pipeline completed."
        ;;
    
    train)
        echo "🤖 Training ML models..."
        docker-compose exec ml-trader python models/ml/train_ml.py
        echo "✅ ML training completed."
        ;;
    
    train-dl)
        echo "🧠 Training Deep Learning models..."
        docker-compose exec ml-trader python models/dl/train_dl.py
        echo "✅ DL training completed."
        ;;
    
    shell)
        echo "🐚 Opening shell in ml-trader container..."
        docker-compose exec ml-trader bash
        ;;
    
    clean)
        echo "🧹 Cleaning up..."
        docker-compose down -v
        docker system prune -af
        echo "✅ Cleanup completed."
        ;;
    
    status)
        echo "📊 Current status:"
        docker-compose ps
        ;;
    
    *)
        echo "Usage: $0 {start|stop|restart|logs|data|train|train-dl|shell|clean|status}"
        echo ""
        echo "Commands:"
        echo "  start     - Start all services"
        echo "  stop      - Stop all services"
        echo "  restart   - Restart all services"
        echo "  logs      - Show logs"
        echo "  data      - Run data pipeline"
        echo "  train     - Train ML models"
        echo "  train-dl  - Train Deep Learning models"
        echo "  shell     - Open shell in container"
        echo "  clean     - Clean up containers and images"
        echo "  status    - Show service status"
        exit 1
        ;;
esac