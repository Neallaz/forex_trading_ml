.PHONY: help build run stop clean test train dashboard jupyter

# Colors
GREEN=\033[0;32m
NC=\033[0m

help:
	@echo "Forex ML Trading System - Docker Management"
	@echo ""
	@echo "Usage:"
	@echo "  make build        Build Docker images"
	@echo "  make run          Start all services"
	@echo "  make run-dev      Start development services"
	@echo "  make stop         Stop all services"
	@echo "  make clean        Remove containers and volumes"
	@echo "  make logs         View logs"
	@echo "  make shell        Open shell in container"
	@echo "  make test         Run tests"
	@echo "  make train        Run training pipeline"
	@echo "  make dashboard    Start only dashboard"
	@echo "  make jupyter      Start Jupyter notebook"
	@echo "  make mlflow       Start MLflow tracking"
	@echo ""

build:
	@echo "${GREEN}Building Docker images...${NC}"
	docker-compose build --no-cache

build-dev:
	@echo "${GREEN}Building development image...${NC}"
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml build

run:
	@echo "${GREEN}Starting all services...${NC}"
	docker-compose up -d

run-dev:
	@echo "${GREEN}Starting development environment...${NC}"
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

stop:
	@echo "${GREEN}Stopping services...${NC}"
	docker-compose down

clean:
	@echo "${GREEN}Cleaning up...${NC}"
	docker-compose down -v --remove-orphans
	docker system prune -f

logs:
	@echo "${GREEN}Viewing logs...${NC}"
	docker-compose logs -f forex-app

shell:
	@echo "${GREEN}Opening shell in container...${NC}"
	docker-compose exec forex-app /bin/bash

test:
	@echo "${GREEN}Running tests...${NC}"
	docker-compose exec forex-app python -m pytest tests/ -v

train:
	@echo "${GREEN}Running training pipeline...${NC}"
	docker-compose exec forex-app python main.py --steps download preprocess features ml dl ensemble

dashboard:
	@echo "${GREEN}Starting dashboard...${NC}"
	docker-compose up -d forex-app
	@echo "Dashboard available at: http://localhost:8501"

jupyter:
	@echo "${GREEN}Starting Jupyter notebook...${NC}"
	docker-compose up -d jupyter
	@echo "Jupyter available at: http://localhost:8888"

mlflow:
	@echo "${GREEN}Starting MLflow...${NC}"
	docker-compose up -d mlflow postgres minio
	@echo "MLflow available at: http://localhost:5000"

update:
	@echo "${GREEN}Updating dependencies...${NC}"
	docker-compose exec forex-app pip install --upgrade -r requirements.txt

backup:
	@echo "${GREEN}Backing up data...${NC}"
	docker-compose exec forex-app tar czf /tmp/backup_$(date +%Y%m%d_%H%M%S).tar.gz /app/data /app/models /app/trading/backtesting/results
	docker cp forex-app:/tmp/backup_*.tar.gz ./backups/

restore:
	@echo "${GREEN}Restoring from backup...${NC}"
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "Usage: make restore BACKUP_FILE=./backups/backup_YYYYMMDD_HHMMSS.tar.gz"; \
		exit 1; \
	fi
	docker cp $(BACKUP_FILE) forex-app:/tmp/restore.tar.gz
	docker-compose exec forex-app tar xzf /tmp/restore.tar.gz -C /

monitor:
	@echo "${GREEN}Monitoring resources...${NC}"
	watch -n 2 'docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"'