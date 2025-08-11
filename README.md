````markdown
# Guide d'utilisation pour le projet RAG-mini

Ce tutoriel pour configurer l'environnement, télécharger le modèle, ingérer des documents, et lancer l'application. 

---

## Prérequis

- **Git** installé (https://git-scm.com/downloads)
- **Python 3.10+** installé (https://www.python.org/downloads)
- **Accès Internet** pour télécharger le modèle Llama 2
- Un **navigateur web** moderne

---

## 1. Cloner le dépôt

1. Ouvrez un terminal (CMD, PowerShell, iTerm, etc.)
2. Exécutez :
   ```bash
   git clone https://github.com/<VOTRE_PSEUDO>/rag-mini.git
   cd rag-mini
````

---

## 2. Créer et activer l'environnement virtuel

### Sous Windows (CMD)

```batch
python -m venv .venv
.venv\Scripts\activate.bat
```

### Sous macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

> Le prompt doit afficher `(.venv)` pour indiquer que l'environnement est actif.

---

## 3. Installer les dépendances Python

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Vérifiez que les paquets suivants sont bien installés :

- streamlit
- langchain (+ langchain-community ou langchain-chroma)
- chromadb
- pypdf
- python-dotenv
- llama-cpp-python
- sentence-transformers
- pytest
- playwright

---

## 4. Télécharger le modèle Llama 2 7B-Chat

Le modèle n'est pas stocké dans Git pour alléger le dépôt.

### Option 1 : Script de téléchargement

- **Windows** (`download_model.bat`)
  ```batch
  mkdir models
  powershell -Command "Invoke-WebRequest -Uri 'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf' -OutFile 'models/llama-2-7b-chat.Q4_K_M.gguf'"
  ```
- **macOS / Linux** (`download_model.sh`)
  ```bash
  mkdir -p models
  curl -L -o models/llama-2-7b-chat.Q4_K_M.gguf \
    https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
  ```

### Option 2 : Téléchargement manuel

1. Allez sur HuggingFace : [https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
2. Téléchargez le fichier `.gguf`
3. Placez-le dans le dossier `models/` à la racine du projet

---

## 5. Préparer vos documents

1. Créez un dossier `data/` à la racine :
   ```bash
   mkdir data
   ```
2. Copiez vos fichiers **.txt** ou **.pdf** dans `data/`

---

## 6. Ingérer les documents

Exécutez l'ingestion en CLI :

```bash
python ingest.py data/mon_document.txt
```

Vous devez voir :

```
✅  Ingestion terminée
```

Le dossier `chroma_db/` est automatiquement créé et rempli d'index binaires.

---

## 7. Lancer l'application Streamlit

```bash
streamlit run app.py
```

- Votre navigateur s'ouvre sur `http://localhost:8501`
- **Upload** de nouveaux fichiers via le bouton "Indexer"
- Posez votre **question** dans la chatbox
- Cochez **Debug** dans la barre latérale pour voir les passages récupérés

---

