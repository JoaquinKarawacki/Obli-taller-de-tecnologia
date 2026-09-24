"""Configuración central del proyecto: rutas, modelos y constantes.

Todo el código del proyecto está en español (ver PLAN.md, sección de convenciones).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Carga las variables de entorno desde .env (por ejemplo, GROQ_API_KEY)
load_dotenv()

# --- Rutas del proyecto ---
RAIZ = Path(__file__).resolve().parent.parent
DIR_DATOS = RAIZ / "data"
DIR_CHROMA = RAIZ / "chroma_db"

# --- Dataset ---
# Corpus individuales que se unen en el dataset consolidado.
# Se excluye "phishing_email.csv" porque ya es una combinación que pierde los metadatos.
ARCHIVOS_CORPUS = [
    "CEAS_08.csv",
    "Enron.csv",
    "Ling.csv",
    "Nazario.csv",
    "Nigerian_Fraud.csv",
    "SpamAssasin.csv",
]
COLUMNAS_RELEVANTES = ["fuente", "sender", "subject", "body", "urls", "label"]
CSV_CONSOLIDADO = DIR_DATOS / "dataset_consolidado.csv"

# --- Modelos ---
MODELO_LLM = "llama-3.3-70b-versatile"          # Groq (gratis, multilingüe)
MODELO_EMBEDDINGS = "intfloat/multilingual-e5-small"  # embeddings locales multilingües

# El modelo e5 requiere prefijos: "passage: " para documentos y "query: " para consultas.
PREFIJO_PASSAGE = "passage: "
PREFIJO_QUERY = "query: "

# --- Ingesta (Fase 2) ---
# Cantidad de correos a indexar (indicación del profe: 10k, no todo el dataset).
# Se toma balanceado: la mitad phishing (label=1) y la mitad legítimos (label=0).
TAMANO_MUESTRA = 10_000
SEMILLA = 42            # para que la muestra sea reproducible
CHUNK_SIZE = 800        # tamaño de cada fragmento (caracteres)
CHUNK_OVERLAP = 100     # solapamiento entre fragmentos

# --- ChromaDB ---
COLECCION_DATASET = "dataset"   # correos del dataset de phishing
COLECCION_PAPERS = "papers"     # documentos largos (papers)

# --- Credenciales ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
