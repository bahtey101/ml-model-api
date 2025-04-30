# Базовый минималистичный образ Python
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем только файл зависимостей сначала (для кеширования)
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения и модель
COPY ./app ./app
COPY ./store ./store

# Указываем порт, который будет слушать приложение
EXPOSE 8000

# Запускаем FastAPI-приложение через uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]