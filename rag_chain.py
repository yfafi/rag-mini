import os
import pathlib
from langchain.embeddings import HuggingFaceEmbeddings, AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Mode local ou Azure
MODE = os.getenv("MODE", "local").lower()
DB_DIR = "./chroma_db"

# ---------- Embeddings ----------
if MODE == "azure":
    embed = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        deployment=os.getenv("AZURE_EMBED_DEPLOYMENT"),
    )
else:
    # embeddings locaux via sentence-transformers
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# ---------- Ingestion ----------
def ingest(file_paths: list[str]):
    """
    Charge et indexe une liste de fichiers texte/PDF :
    - découpe en chunks
    - calcule embeddings
    - stocke dans Chroma
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1_000, chunk_overlap=150)
    docs = []
    for path in file_paths:
        # lit tout le texte, ignore les erreurs d'encodage
        text = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
        # découpe en documents
        docs += splitter.create_documents([text])
    # construit ou recharge la base Chroma
    vectordb = Chroma.from_documents(
        docs,
        embedding=embed,
        persist_directory=DB_DIR,
        collection_name="rag"
    )
    vectordb.persist()

# ---------- Chargement du LLM ----------
def _load_llm():
    if MODE == "azure":
        from langchain.chat_models import AzureChatOpenAI
        return AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_GPT_DEPLOYMENT"),
            temperature=0
        )
    else:
        from langchain.llms import LlamaCpp
        # chemin vers ton modèle local
        llm_path = "./models/llama-2-7b-chat.Q4_K_M.gguf"
        return LlamaCpp(
            model_path=llm_path,
            n_ctx=4096,
            temperature=0,
            n_threads=os.cpu_count()
        )

# ---------- Construction de la chaîne QA ----------
def build_chain():
    """
    Retourne un objet RetrievalQA prêt à l’emploi
    """
    vectordb = Chroma(
        embedding_function=embed,
        persist_directory=DB_DIR,
        collection_name="rag"
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate(
        template=(
            "Tu es un assistant et tu dois répondre STRICTEMENT avec le CONTEXTE.\n"
            "CONTEXTE:\n{context}\n\n"
            "Question: {question}\nRéponse concise en français:"
        ),
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=_load_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
