FROM python:3.12.3-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем необходимые инструменты для управления пакетами
RUN apt-get update && apt-get install -y \
    wget \
    gnupg2 \
    apt-transport-https \
    ca-certificates \
    g++ \
    libxml2-dev \
    libcairo2-dev \
    libfreetype6-dev \
    libfribidi-dev \
    libharfbuzz-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libfontconfig1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Скопируем requirements.txt в контейнер
COPY requirements.txt .

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Скопируем все файлы проекта в контейнер
COPY . .

COPY . /app

COPY creditdefault.csv /app/creditdefault.csv

# Запускаем Python-скрипт
CMD ["python", "mlops_hw1.py"]




