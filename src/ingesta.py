"""Fase 2 — Ingesta y embeddings.

Corta los textos en fragmentos (chunking), genera embeddings multilingües y los
almacena en ChromaDB (colecciones ``dataset`` y ``papers``).
"""
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from src import config


def construir_indice_dataset():
    """Genera embeddings del dataset consolidado y los guarda en Chroma."""
    # TODO Fase 2: chunking + embeddings + Chroma (colección dataset)
    raise NotImplementedError


def construir_indice_papers(rutas_pdf):
    """Procesa papers (PDF), los chunkea y los guarda en Chroma."""
    # TODO Fase 2: leer PDFs + chunking + embeddings + Chroma (colección papers)
    raise NotImplementedError
