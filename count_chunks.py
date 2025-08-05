from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os

# adapte si tu es en MODE=azure
embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embed, collection_name="rag")
stats = db._collection.count()  # nombre total de vecteurs
print("Chunks ingérés :", stats)
