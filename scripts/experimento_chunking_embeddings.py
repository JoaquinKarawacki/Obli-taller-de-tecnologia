"""Experimento opcional: estrategias de chunking × modelos de embeddings.

Cubre dos opcionales de la letra ("comparar diferentes modelos de embeddings" y
"probar distintas estrategias de chunking y recuperación"). Parte de la Parte 2
del Lab03 (comparar chunks pequeños / medianos / grandes), pero en vez de solo
contar chunks MIDE LA CALIDAD de la recuperación:

    Para cada pregunta se sabe qué frase del paper contiene la respuesta. Hay
    ACIERTO si alguno de los k fragmentos recuperados contiene esa frase.

Métricas:
  - acierto@k: % de preguntas con la respuesta entre los k recuperados.
  - MRR: promedio de 1/posición del primer fragmento correcto (1 = siempre primero).

Cada pregunta se hace en español y en inglés para medir el retrieval
cross-lingual (los papers están en inglés y el chatbot debe aceptar español).

Para que corra en pocos minutos se usan los 4 papers cortos (sin GPT-3 ni el
libro de Kleppmann) y colecciones de Chroma en memoria (no toca chroma_db/).

Uso (desde la raíz del repo, con el venv):
    python -m scripts.experimento_chunking_embeddings
"""
import time

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.ingesta import EmbeddingsE5, es_fragmento_util, normalizar_texto

K = 4

PAPERS = [
    "Attention is all you need.pdf",
    "GPT Improving Language Understanding by Generative Pretraining.pdf",
    "Scaling Laws for Neural Language Models.pdf",
    "2507.14805v1.pdf",
]

# (pregunta en español, pregunta en inglés, frase que contiene la respuesta)
PREGUNTAS = [
    ("¿Cuántas cabezas de atención en paralelo usa el Transformer?",
     "How many parallel attention heads does the Transformer use?", "parallel attention layers"),
    ("¿Qué optimizador se usó para entrenar el Transformer?",
     "Which optimizer was used to train the Transformer?", "Adam optimizer"),
    ("¿Qué técnica de regularización suaviza las etiquetas en el Transformer?",
     "What regularization smooths the target labels in the Transformer?", "label smoothing"),
    ("¿Con qué corpus de libros se pre-entrenó GPT?",
     "Which book corpus was GPT pre-trained on?", "BooksCorpus"),
    ("¿GPT usa un objetivo auxiliar de modelado de lenguaje durante el fine-tuning?",
     "Does GPT use an auxiliary language modeling objective during fine-tuning?", "auxiliary objective"),
    ("¿Cuánto influye la forma del modelo (profundidad vs ancho) en el rendimiento?",
     "How much does model shape (depth vs width) affect performance?", "weakly on model shape"),
    ("¿Cuándo empieza el sobreajuste según las leyes de escalado?",
     "When does overfitting start according to the scaling laws?", "overfitting"),
    ("¿El aprendizaje subliminal funciona si el maestro y el alumno tienen distinto modelo base?",
     "Does subliminal learning work when teacher and student have a different base model?", "different base model"),
    ("¿Se observa aprendizaje subliminal en un clasificador de dígitos?",
     "Is subliminal learning observed in an MNIST digit classifier?", "MNIST"),
]

ESTRATEGIAS_CHUNKING = {  # (chunk_size, chunk_overlap); las 3 primeras son las del Lab03
    "pequeños 250/30": (250, 30),
    "medianos 600/100": (600, 100),
    "grandes 1200/200": (1200, 200),
    "proyecto 800/100": (config.CHUNK_SIZE, config.CHUNK_OVERLAP),
}

MODELOS_EMBEDDINGS = {
    "all-MiniLM-L6-v2 (Lab02/03, inglés)": lambda: HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    "multilingual-e5-small (proyecto)": lambda: EmbeddingsE5(model_name=config.MODELO_EMBEDDINGS),
}


def cargar_paginas():
    """Carga los papers página por página, igual que la ingesta del proyecto."""
    paginas = []
    for archivo in PAPERS:
        for pagina in PyPDFLoader(str(config.DIR_PAPERS / archivo)).load():
            pagina.page_content = normalizar_texto(pagina.page_content)
            if len(pagina.page_content) >= 50:
                paginas.append(pagina)
    return paginas


def evaluar(indice: Chroma, idioma: int) -> tuple[float, float]:
    """Devuelve (acierto@K, MRR) para las preguntas en el idioma dado (0=ES, 1=EN)."""
    aciertos, rr = 0, 0.0
    for pregunta in PREGUNTAS:
        clave = pregunta[2].lower()
        resultados = indice.similarity_search(pregunta[idioma], k=K)
        posicion = next((i for i, doc in enumerate(resultados, 1) if clave in doc.page_content.lower()), None)
        if posicion:
            aciertos += 1
            rr += 1 / posicion
    return aciertos / len(PREGUNTAS), rr / len(PREGUNTAS)


if __name__ == "__main__":
    paginas = cargar_paginas()
    filas = []
    for nombre_modelo, crear_modelo in MODELOS_EMBEDDINGS.items():
        modelo = crear_modelo()
        for nombre_estrategia, (tamano, solapamiento) in ESTRATEGIAS_CHUNKING.items():
            separador = RecursiveCharacterTextSplitter(chunk_size=tamano, chunk_overlap=solapamiento)
            fragmentos = [f for f in separador.split_documents(paginas) if es_fragmento_util(f.page_content)]

            inicio = time.time()
            indice = Chroma(collection_name=f"exp_{len(filas)}", embedding_function=modelo)  # en memoria
            indice.add_documents(fragmentos)
            segundos = time.time() - inicio

            acierto_es, mrr_es = evaluar(indice, 0)
            acierto_en, mrr_en = evaluar(indice, 1)
            filas.append((nombre_modelo, nombre_estrategia, len(fragmentos), segundos,
                          acierto_es, mrr_es, acierto_en, mrr_en))
            print(f"{nombre_modelo} | {nombre_estrategia}: ES {acierto_es:.0%} / EN {acierto_en:.0%} ({segundos:.0f} s)")
            indice.delete_collection()

    # Tabla en Markdown para pegar en la BITACORA / informe.
    print(f"\n| Embeddings | Chunking | Fragmentos | Indexado (s) | Acierto@{K} ES | MRR ES | Acierto@{K} EN | MRR EN |")
    print("|---|---|---|---|---|---|---|---|")
    for m, e, n, s, a_es, r_es, a_en, r_en in filas:
        print(f"| {m} | {e} | {n} | {s:.0f} | {a_es:.0%} | {r_es:.2f} | {a_en:.0%} | {r_en:.2f} |")
