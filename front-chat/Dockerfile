FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de requerimientos
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip3 install -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Asegurarse de que los archivos tengan los permisos correctos
RUN chmod -R 755 /app

# Crear directorio para logs
RUN mkdir -p /tmp && chmod 777 /tmp

# Variables de entorno por defecto
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Exponer el puerto
EXPOSE 8501

# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
