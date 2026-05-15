# 1. Image de base légère (Debian Slim)
FROM python:3.11-slim

# 2. Configuration de l'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=2.1.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# 3. Installation des dépendances système (CRITIQUE POUR LE MACHINE LEARNING)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Installation de Poetry
RUN pip install "poetry==$POETRY_VERSION"

# 5. Définition du répertoire de travail
WORKDIR /app

# 6. Installation des dépendances Python (Méthode forte)
# 6. Installation des dépendances Python
COPY pyproject.toml ./
RUN poetry install --no-root

# 7. Copie du code source et des artefacts
COPY api/ ./api/
COPY src/ ./src/
COPY model.pkl ./api/
COPY expected_features.json ./api/

# 8. Port exposé par l'API
EXPOSE 8000

# 9. Commande de lancement (Gradio)
CMD ["python", "api/main.py"]