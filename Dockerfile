# Usar imagen oficial de Python
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
# IMPORTANTE: Asegúrate de que el archivo se llame 'main.py' en tu proyecto
COPY main.py .

# Configurar variables de entorno
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Exponer puerto
EXPOSE 8080

# Usar Gunicorn como servidor de producción
# Configuración optimizada:
# - workers: 2 (ajustar según CPU disponible)
# - threads: 4 (para I/O concurrente)
# - timeout: 300 (5 minutos para el análisis inicial)
# - worker-class: gthread (para aprovechar threading)
CMD exec gunicorn --bind :$PORT --workers 2 --threads 4 --timeout 300 --worker-class gthread main:app
