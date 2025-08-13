# Image de base
FROM python:3.11-slim

# Bonnes pratiques Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# (optionnel) outils de build si jamais une lib en a besoin
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Utilisateur non-root
RUN useradd -m appuser
WORKDIR /app

# Dépendances
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Code
COPY . .

# Dossiers de travail + droits
RUN mkdir -p /app/chroma_db /app/data && chown -R appuser:appuser /app
USER appuser

# Streamlit
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
