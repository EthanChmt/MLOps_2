# 1. Image de base légère (Debian Slim)
FROM python:3.11-slim

# 2. Configuration de l'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=1.7.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# 3. Installation de Poetry et des dépendances système minimales
RUN pip install "poetry==$POETRY_VERSION"

# 4. Définition du répertoire de travail
WORKDIR /app

# 5. Installation des dépendances (Etape séparée pour optimiser le cache Docker)
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --without dev

# 6. Copie du code source et du modèle
COPY api/ ./api/
COPY src/ ./src/
COPY model_lightgbm.pkl .

# 7. Port exposé par l'API
EXPOSE 8000

# 8. Commande de lancement (adapter selon si tu utilises FastAPI ou Gradio)
# Pour FastAPI :
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]