# Makefile
SHELL := /bin/bash
.ONESHELL:

PROJECT_NAME=ml-model-api
CONTAINER_NAME=ml-model-service
MODEL_VOLUME=ml_model_data
PORT=8000

.PHONY: venv install run-local build run stop logs

.PHONY: venv install run-local build run stop logs

# 1) Создание виртуального окружения
install:
	@echo "Creating virtualenv and installing dependencies..."
	python3 -m venv venv && \
	source venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt

# 3) Локальный запуск сервиса с автоперезагрузкой
run-local: install
	@echo "Running FastAPI app locally..."
	@source venv/bin/activate && uvicorn app.main:app --reload

# 4) Сборка Docker-образа
build:
	docker build -t $(PROJECT_NAME) .

# 5) Запуск Docker-контейнера с монтированием модели
run:
	docker run -d -p $(PORT):8000 \
		-v $$HOME/$(MODEL_VOLUME):/app/model_store \
		--name $(CONTAINER_NAME) $(PROJECT_NAME)

# 6) Остановка и удаление контейнера
stop:
	docker stop $(CONTAINER_NAME) && docker rm $(CONTAINER_NAME)

# 7) Просмотр логов контейнера
logs:
	docker logs -f $(CONTAINER_NAME)