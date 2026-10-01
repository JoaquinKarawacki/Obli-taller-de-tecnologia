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

# --- Papers (documentos largos) ---
# Los PDFs de la materia se copian a data/papers/ (no se commitean).
DIR_PAPERS = DIR_DATOS / "papers"
# Título legible de cada paper (por nombre de archivo). Se guarda como metadato en
# cada fragmento: sirve para citar la fuente y para filtrar la búsqueda por paper.
# Si aparece un PDF que no está acá, se usa el nombre del archivo como título.
TITULOS_PAPERS = {
    "2507.14805v1.pdf": "Subliminal Learning: Language models transmit behavioral traits via hidden signals in data",
    "Attention is all you need.pdf": "Attention Is All You Need",
    "Designing Data-Intensive Applications The Big Ideas Behind Reliable, Scalable, and Maintainable Systems by Martin Kleppmann (z-l.pdf": "Designing Data-Intensive Applications (Martin Kleppmann)",
    "GPT Improving Language Understanding by Generative Pretraining.pdf": "Improving Language Understanding by Generative Pre-Training (GPT)",
    "Language Models are Few-Shot Learners (GPT3).pdf": "Language Models are Few-Shot Learners (GPT-3)",
    "Scaling Laws for Neural Language Models.pdf": "Scaling Laws for Neural Language Models",
}

# Proporción mínima de letras para indexar un fragmento de paper (descarta tablas
# de números, que contaminan la búsqueda; ver ingesta.es_fragmento_util).
PROPORCION_MIN_LETRAS = 0.5

# --- Recuperación (Fase 3) ---
K_FRAGMENTOS = 4        # cantidad de fragmentos que se recuperan por búsqueda
# Relevancia mínima (0 a 1) para considerar que un fragmento es evidencia válida.
# Si ningún fragmento la supera, la búsqueda devuelve "sin evidencia" y el agente
# debe admitir que no tiene información (control de alucinaciones).
# Calibrado con scripts/probar_recuperacion.py: las consultas pertinentes dieron
# 0.72-0.83 y las ajenas 0.63-0.75 (e5 "comprime" los puntajes y se solapan).
# Por eso es solo un PISO que corta lo claramente ajeno; el resto lo filtran el
# router (small-talk no busca) y el prompt del generador (verificar evidencia).
UMBRAL_RELEVANCIA = 0.68

# --- ChromaDB ---
COLECCION_DATASET = "dataset"   # correos del dataset de phishing
COLECCION_PAPERS = "papers"     # documentos largos (papers)

# --- Credenciales ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
