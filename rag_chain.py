import os
import pathlib
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_openai import AzureOpenAIEmbeddings 


# --------- Config ---------
DB_DIR = "./chroma_db"  # dossier où Chroma persiste l'index


# --------- Helpers ---------
def _require_env(name: str) -> str:
    """Récupère une variable d'env ou lève une erreur claire."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Variable d'environnement manquante: {name}.\n"
            f"Exemple (Windows CMD):   set {name}=<valeur>\n"
            f"Exemple (PowerShell):     $Env:{name}=\"<valeur>\"\n"
        )
    return value


def _read_file_text(path: str) -> str:
    """Lit un .txt ou extrait le texte d'un .pdf (si dispo), sinon lit en brut."""
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")

    if p.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader  # nécessite `pypdf`
            reader = PdfReader(str(p))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            print(f"[WARN] Extraction PDF échouée pour {path}: {e}. Lecture brute UTF-8.")
            return p.read_text(encoding="utf-8", errors="ignore")
    else:
        return p.read_text(encoding="utf-8", errors="ignore")


# --------- Embeddings (Local) ---------
# Modèle robuste pour RAG (768-dim), pas de quota Azure requis
# embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# ------------ Embeddings Azure (Modèle text-embedding-ada-002) --------- 

embed = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_EMBED_DEPLOYMENT"),
    api_version="2024-12-01-preview",
)

# --------- Ingestion ---------
def ingest(file_paths: List[str]) -> None:
    """
    Ingestion RAG :
      - lecture fichiers (txt/pdf)
      - découpe en chunks
      - embeddings via HuggingFace
      - stockage dans Chroma (persist_directory=DB_DIR)
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    docs: List[Document] = []
    for path in file_paths:
        text = _read_file_text(path)
        # on garde la source pour l'affichage debug
        chunk_docs = splitter.create_documents([text], metadatas=[{"source": path}])
        docs.extend(chunk_docs)

    # Crée/actualise l’index Chroma (persisté sur disque)
    Chroma.from_documents(
        docs,
        embed,
        persist_directory=DB_DIR,
        collection_name="rag",
    )
    print("✅  Ingestion terminée (index Chroma mis à jour).")


# --------- LLM (Azure OpenAI) ---------
def _load_llm() -> AzureChatOpenAI:
    """
    Retourne un chat model Azure OpenAI.
    ATTEND les variables d'env :
      - AZURE_OPENAI_API_BASE  (ex: https://<nom>.openai.azure.com/)
      - AZURE_OPENAI_API_KEY   (clé 1)
      - AZURE_GPT_DEPLOYMENT   (ex: gpt4o)
    """
    azure_endpoint = _require_env("AZURE_OPENAI_API_BASE")
    api_key = _require_env("AZURE_OPENAI_API_KEY")
    deployment = _require_env("AZURE_GPT_DEPLOYMENT")

    return AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        azure_deployment=deployment,
        api_version="2024-12-01-preview",
        temperature=0,
    )


# --------- Chaîne QA ---------
def build_chain() -> RetrievalQA:
    """
    Construit une chaîne RetrievalQA :
      - Retriever = Chroma (embeddings HF)
      - LLM = AzureChatOpenAI
      - Prompt strictement basé sur le contexte
    """
    vectordb = Chroma(
        embedding_function=embed,
        persist_directory=DB_DIR,
        collection_name="rag",
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate(
        template=(
            "Tu es un assistant et tu dois répondre STRICTEMENT avec le CONTEXTE fourni.\n"
            "Si l'information n'est pas dans le contexte, dis-le explicitement.\n\n"
            "CONTEXTE:\n{context}\n\n"
            "Question: {question}\n"
            "Réponse concise en français:"
        ),
        input_variables=["context", "question"],
    )

    return RetrievalQA.from_chain_type(
        llm=_load_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
