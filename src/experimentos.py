"""Opcionales — experimentos de recuperación (sin LLM).

Cubre dos opcionales de la letra:
  - "Probar distintas estrategias de chunking y recuperación de información".
  - "Comparar diferentes modelos de embeddings".

Idea general: medir la CALIDAD de la recuperación, no solo contar fragmentos.
Para eso se arma un conjunto de preguntas sobre los papers en el que se sabe de
antemano en qué página está la respuesta (se verificó leyendo cada PDF). Cada
configuración (chunking, forma de recuperar o modelo de embeddings) se evalúa
con las mismas preguntas:

  - acierto@k: % de preguntas en las que al menos uno de los k fragmentos
    recuperados viene de una página correcta.
  - MRR (Mean Reciprocal Rank): promedio de 1/posición del primer fragmento
    correcto. 1 = siempre primero; 0,5 = en promedio segundo; 0 = nunca aparece.

Cada pregunta está en español y en inglés: los papers están en inglés y el
chatbot tiene que entender preguntas en español (recuperación entre idiomas).

Para la comparación de embeddings se mide además cuánto se separan los puntajes
de las preguntas pertinentes y los de preguntas ajenas a los papers. Eso decide
si un umbral de relevancia (``UMBRAL_RELEVANCIA``) puede cortar lo ajeno sin
perder lo pertinente.

Decisiones para que los experimentos sean comparables y corran en minutos:
  - Se usan los 5 papers (sin el libro de Kleppmann, que tiene 600 páginas y
    haría cada configuración 4 veces más lenta).
  - Colecciones de Chroma EN MEMORIA: no se toca ``chroma_db/``.
  - Mismo preprocesamiento que la ingesta del proyecto (``normalizar_texto`` y
    ``es_fragmento_util``) y fragmentación página por página, así cada fragmento
    conserva su número de página.
  - Todos los modelos de embeddings devuelven vectores normalizados y Chroma usa
    la distancia L2, igual que el proyecto. Con vectores normalizados, L2 y
    coseno ordenan igual (L2² = 2 − 2·coseno) y el puntaje de relevancia queda en
    la misma escala que el umbral del proyecto.
  - No se usa el LLM: no gasta cuota de Groq y el resultado es determinista.
"""
import re
import time
import uuid
from dataclasses import dataclass

import numpy as np
import pandas as pd
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.ingesta import EmbeddingsE5, es_fragmento_util, normalizar_texto

K = config.K_FRAGMENTOS  # 4, el mismo k que usa el agente

ATTENTION = "Attention is all you need.pdf"
GPT = "GPT Improving Language Understanding by Generative Pretraining.pdf"
GPT3 = "Language Models are Few-Shot Learners (GPT3).pdf"
SCALING = "Scaling Laws for Neural Language Models.pdf"
SUBLIMINAL = "2507.14805v1.pdf"
PAPERS = [ATTENTION, GPT, GPT3, SCALING, SUBLIMINAL]


@dataclass(frozen=True)
class Pregunta:
    """Pregunta de evaluación con su respuesta ubicada en el paper."""

    es: str
    en: str
    archivo: str
    paginas: tuple[int, ...]  # páginas (desde 1) donde está la respuesta


PREGUNTAS = [
    # --- Attention Is All You Need ---
    Pregunta("¿Cuántas cabezas de atención en paralelo usa el Transformer?",
             "How many parallel attention heads does the Transformer use?", ATTENTION, (5,)),
    Pregunta("¿Qué optimizador se usó para entrenar el Transformer?",
             "Which optimizer was used to train the Transformer?", ATTENTION, (7,)),
    Pregunta("¿Qué valor de label smoothing se usó al entrenar el Transformer?",
             "What label smoothing value was used when training the Transformer?", ATTENTION, (8,)),
    Pregunta("¿Cuántos pares de oraciones tiene el dataset WMT 2014 inglés-alemán?",
             "How many sentence pairs are in the WMT 2014 English-German dataset?", ATTENTION, (7,)),
    Pregunta("¿Cómo codifica el Transformer la posición de las palabras?",
             "How does the Transformer encode the position of the tokens?", ATTENTION, (6,)),
    Pregunta("¿Qué puntaje BLEU obtuvo el Transformer en inglés-alemán?",
             "What BLEU score did the Transformer achieve on English-to-German?", ATTENTION, (1, 8)),
    # --- GPT (Improving Language Understanding) ---
    Pregunta("¿Con qué corpus de libros se pre-entrenó GPT?",
             "Which book corpus was GPT pre-trained on?", GPT, (4, 5)),
    Pregunta("¿Cuántas capas y cabezas de atención tiene el modelo GPT?",
             "How many layers and attention heads does the GPT model have?", GPT, (5,)),
    Pregunta("¿Ayuda el objetivo auxiliar de modelado de lenguaje durante el fine-tuning de GPT?",
             "Does the auxiliary language modeling objective help during GPT fine-tuning?", GPT, (8,)),
    Pregunta("¿Cómo adapta GPT las entradas estructuradas a cada tarea?",
             "How does GPT adapt structured inputs for each task?", GPT, (4,)),
    # --- GPT-3 (Language Models are Few-Shot Learners) ---
    Pregunta("¿Cuál es el tamaño de la ventana de contexto de GPT-3?",
             "What is the context window size of GPT-3?", GPT3, (6, 8)),
    Pregunta("¿Qué tan bien distinguen las personas noticias escritas por GPT-3?",
             "How well can humans tell apart news articles written by GPT-3?", GPT3, (26, 27)),
    Pregunta("¿Cuánto cómputo total se usó para entrenar GPT-3 175B?",
             "How much total compute was used to train GPT-3 175B?", GPT3, (46,)),
    Pregunta("¿Qué es la contaminación de datos y por qué preocupa en GPT-3?",
             "What is data contamination and why is it a concern for GPT-3?", GPT3, (6, 29, 30, 31)),
    # --- Scaling Laws ---
    Pregunta("¿Cuánto influye la forma del modelo (profundidad vs. ancho) en el rendimiento?",
             "How much does model shape (depth vs width) affect performance?", SCALING, (3, 5)),
    Pregunta("¿Cuánto hay que aumentar los datos al aumentar el tamaño del modelo para evitar el sobreajuste?",
             "How much must the data grow when the model grows to avoid overfitting?", SCALING, (3, 5)),
    Pregunta("¿Conviene entrenar los modelos hasta la convergencia con un presupuesto fijo de cómputo?",
             "Should models be trained to convergence under a fixed compute budget?", SCALING, (1, 3)),
    # --- Subliminal Learning ---
    Pregunta("¿El aprendizaje subliminal funciona si maestro y alumno tienen distinto modelo base?",
             "Does subliminal learning work when teacher and student have different base models?", SUBLIMINAL, (1, 2)),
    Pregunta("¿Se observa aprendizaje subliminal en un clasificador de MNIST?",
             "Is subliminal learning observed in an MNIST classifier?", SUBLIMINAL, (2, 11)),
    Pregunta("¿Qué demuestra el teorema sobre un paso de descenso por gradiente?",
             "What does the theorem about a single gradient descent step show?", SUBLIMINAL, (2, 10)),
]

# Preguntas que NO tienen respuesta en los papers: sirven para medir si el
# puntaje de relevancia permite distinguir lo pertinente de lo ajeno.
CONSULTAS_AJENAS = [
    "¿Cuál es la receta de la lasaña?",
    "¿Cuáles son las mejores playas de Uruguay?",
    "¿Quién ganó el Mundial de fútbol de 1950?",
    "¿Cómo se cambia una rueda pinchada?",
    "¿Cuál es la capital de Australia?",
    "What are the symptoms of the flu?",
    "How do I bake sourdough bread?",
    "Who painted the Mona Lisa?",
]


# ---------------------------------------------------------------------------
# Carga y estrategias de chunking
# ---------------------------------------------------------------------------


def cargar_paginas(archivos: list[str] = PAPERS) -> list[Document]:
    """Carga los papers página por página, igual que la ingesta del proyecto."""
    paginas = []
    for archivo in archivos:
        titulo = config.TITULOS_PAPERS.get(archivo, archivo)
        for pagina in PyPDFLoader(str(config.DIR_PAPERS / archivo)).load():
            texto = normalizar_texto(pagina.page_content)
            if len(texto) >= 50:
                paginas.append(Document(
                    page_content=texto,
                    metadata={"titulo": titulo, "archivo": archivo, "pagina": pagina.metadata["page"] + 1},
                ))
    return paginas


def _oraciones(texto: str) -> list[str]:
    """Separa un texto en oraciones (punto, signo de pregunta o exclamación + mayúscula)."""
    return [o for o in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])", texto) if o.strip()]


def chunking_recursivo(paginas: list[Document], tamano: int, solapamiento: int) -> list[Document]:
    """Corte recursivo por tamaño (el del proyecto): intenta cortar por párrafo,
    después por línea, por espacio y por último por carácter."""
    separador = RecursiveCharacterTextSplitter(chunk_size=tamano, chunk_overlap=solapamiento)
    return separador.split_documents(paginas)


def chunking_por_oraciones(paginas: list[Document], tamano: int = 800) -> list[Document]:
    """Agrupa oraciones completas hasta llegar a ``tamano`` caracteres.

    A diferencia del corte recursivo, nunca corta una oración al medio, pero no
    tiene solapamiento.
    """
    fragmentos = []
    for pagina in paginas:
        actual = ""
        for oracion in _oraciones(pagina.page_content):
            if actual and len(actual) + len(oracion) + 1 > tamano:
                fragmentos.append(Document(page_content=actual, metadata=pagina.metadata))
                actual = ""
            actual = f"{actual} {oracion}".strip()
        if actual:
            fragmentos.append(Document(page_content=actual, metadata=pagina.metadata))
    return fragmentos


def chunking_semantico(paginas: list[Document], modelo, percentil: int = 20, tamano_max: int = 1500) -> list[Document]:
    """Corta donde cambia el tema.

    Se calcula el embedding de cada oración y la similitud coseno entre oraciones
    consecutivas. Donde la similitud cae por debajo del ``percentil`` (las caídas
    más fuertes del documento) se empieza un fragmento nuevo. Si un fragmento
    queda más largo que ``tamano_max`` se vuelve a cortar con el método
    recursivo, para no superar el largo que el modelo puede leer.
    """
    oraciones_por_pagina = [_oraciones(p.page_content) for p in paginas]
    todas = [o for oraciones in oraciones_por_pagina for o in oraciones]
    vectores = np.array(modelo.embed_documents(todas))

    # Similitud entre cada oración y la siguiente (los vectores ya están normalizados).
    similitudes, posicion = [], 0
    for oraciones in oraciones_por_pagina:
        bloque = vectores[posicion:posicion + len(oraciones)]
        similitudes.append((bloque[:-1] * bloque[1:]).sum(axis=1))
        posicion += len(oraciones)
    corte = np.percentile(np.concatenate([s for s in similitudes if len(s)]), percentil)

    recorte = RecursiveCharacterTextSplitter(chunk_size=tamano_max, chunk_overlap=100)
    fragmentos = []
    for pagina, oraciones, sims in zip(paginas, oraciones_por_pagina, similitudes):
        grupo = []
        for i, oracion in enumerate(oraciones):
            grupo.append(oracion)
            fin_de_tema = i < len(sims) and sims[i] < corte
            if fin_de_tema or i == len(oraciones) - 1:
                for texto in recorte.split_text(" ".join(grupo)):
                    fragmentos.append(Document(page_content=texto, metadata=pagina.metadata))
                grupo = []
    return fragmentos


# ---------------------------------------------------------------------------
# Modelos de embeddings
# ---------------------------------------------------------------------------

# nombre -> (modelo de Hugging Face, usa prefijos de e5, comentario)
MODELOS_EMBEDDINGS = {
    "all-MiniLM-L6-v2": ("sentence-transformers/all-MiniLM-L6-v2", False, "el de los labs, entrenado en inglés"),
    "paraphrase-multilingual-MiniLM-L12-v2": (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", False, "multilingüe, de paráfrasis"),
    "multilingual-e5-small": ("intfloat/multilingual-e5-small", True, "el del proyecto"),
    "multilingual-e5-base": ("intfloat/multilingual-e5-base", True, "e5 más grande"),
}


def crear_modelo(nombre: str):
    """Crea el modelo de embeddings, siempre con vectores normalizados.

    Los modelos e5 necesitan los prefijos "query: " y "passage: " (``EmbeddingsE5``);
    los otros no.
    """
    modelo, usa_prefijos, _ = MODELOS_EMBEDDINGS[nombre]
    clase = EmbeddingsE5 if usa_prefijos else HuggingFaceEmbeddings
    return clase(model_name=modelo, encode_kwargs={"normalize_embeddings": True})


# ---------------------------------------------------------------------------
# Indexación y evaluación
# ---------------------------------------------------------------------------


def indexar(fragmentos: list[Document], modelo) -> tuple[Chroma, float]:
    """Indexa los fragmentos útiles en una colección de Chroma en memoria.

    Devuelve el índice y los segundos que tardó (calcular embeddings + insertar).
    """
    fragmentos = [f for f in fragmentos if es_fragmento_util(f.page_content)]
    inicio = time.time()
    indice = Chroma(collection_name=f"exp_{uuid.uuid4().hex[:8]}", embedding_function=modelo)
    for i in range(0, len(fragmentos), 500):
        indice.add_documents(fragmentos[i:i + 500])
    return indice, time.time() - inicio


def _es_correcto(documento: Document, pregunta: Pregunta) -> bool:
    return documento.metadata["archivo"] == pregunta.archivo and documento.metadata["pagina"] in pregunta.paginas


def recuperar(indice: Chroma, consulta: str, modo: str, pregunta: Pregunta | None = None) -> list[Document]:
    """Recupera fragmentos según el modo de recuperación.

    - ``topk``: los K más parecidos (lo que hace el agente).
    - ``topk8``: los 8 más parecidos.
    - ``umbral``: los K más parecidos, descartando los que no llegan a
      ``UMBRAL_RELEVANCIA`` (lo que hace la tool del proyecto).
    - ``mmr``: Maximal Marginal Relevance; de los 20 más parecidos elige K que
      sean relevantes pero distintos entre sí, para no repetir información.
    - ``filtro``: los K más parecidos, buscando solo dentro del paper correcto
      (simula cuando el agente pasa ``titulo_paper`` a la tool).
    """
    if modo == "topk":
        return indice.similarity_search(consulta, k=K)
    if modo == "topk8":
        return indice.similarity_search(consulta, k=8)
    if modo == "umbral":
        resultados = indice.similarity_search_with_relevance_scores(consulta, k=K)
        return [doc for doc, puntaje in resultados if puntaje >= config.UMBRAL_RELEVANCIA]
    if modo == "mmr":
        return indice.max_marginal_relevance_search(consulta, k=K, fetch_k=20, lambda_mult=0.5)
    if modo == "filtro":
        return indice.similarity_search(consulta, k=K, filter={"archivo": pregunta.archivo})
    raise ValueError(f"Modo de recuperación desconocido: {modo}")


def evaluar(indice: Chroma, modo: str = "topk") -> dict:
    """Calcula acierto@k y MRR en español y en inglés."""
    metricas = {}
    for idioma in ("es", "en"):
        aciertos, rr = 0, 0.0
        for pregunta in PREGUNTAS:
            documentos = recuperar(indice, getattr(pregunta, idioma), modo, pregunta)
            posicion = next((i for i, d in enumerate(documentos, 1) if _es_correcto(d, pregunta)), None)
            if posicion:
                aciertos += 1
                rr += 1 / posicion
        metricas[f"acierto {idioma.upper()}"] = aciertos / len(PREGUNTAS)
        metricas[f"MRR {idioma.upper()}"] = rr / len(PREGUNTAS)
    return metricas


def separacion(indice: Chroma) -> dict:
    """Compara el mejor puntaje de las preguntas pertinentes con el de las ajenas.

    - ``pertinentes`` / ``ajenas``: promedio del puntaje del mejor fragmento.
    - ``separables``: % de pares (pertinente, ajena) en los que la pertinente
      tiene mayor puntaje. 100% = existe un umbral que separa todo; 50% = el
      puntaje no distingue nada.
    - ``ajenas cortadas`` / ``pertinentes perdidas``: qué pasa con el umbral
      actual del proyecto (0,68). Solo tiene sentido para e5-small, el modelo con
      el que se calibró.
    """
    def mejor(consulta):
        return indice.similarity_search_with_relevance_scores(consulta, k=1)[0][1]

    pertinentes = np.array([mejor(getattr(p, idioma)) for p in PREGUNTAS for idioma in ("es", "en")])
    ajenas = np.array([mejor(c) for c in CONSULTAS_AJENAS])
    return {
        "puntaje pertinentes": pertinentes.mean(),
        "puntaje ajenas": ajenas.mean(),
        "separables": (pertinentes[:, None] > ajenas[None, :]).mean(),
        "ajenas cortadas": (ajenas < config.UMBRAL_RELEVANCIA).mean(),
        "pertinentes perdidas": (pertinentes < config.UMBRAL_RELEVANCIA).mean(),
    }


# ---------------------------------------------------------------------------
# Experimentos
# ---------------------------------------------------------------------------



def experimento_chunking(paginas: list[Document], modelo) -> pd.DataFrame:
    """Compara estrategias de chunking con el mismo modelo y recuperación top-k."""
    estrategias = {
        "recursivo 250/30": lambda: chunking_recursivo(paginas, 250, 30),
        "recursivo 500/50": lambda: chunking_recursivo(paginas, 500, 50),
        "recursivo 800/100 (proyecto)": lambda: chunking_recursivo(paginas, 800, 100),
        "recursivo 800/0 (sin solapamiento)": lambda: chunking_recursivo(paginas, 800, 0),
        "recursivo 1200/200": lambda: chunking_recursivo(paginas, 1200, 200),
        "por oraciones (hasta 800)": lambda: chunking_por_oraciones(paginas, 800),
        "semántico (e5, percentil 20)": lambda: chunking_semantico(paginas, modelo),
    }
    filas = []
    for nombre, fragmentar in estrategias.items():
        fragmentos = fragmentar()
        indice, segundos = indexar(fragmentos, modelo)
        largos = [len(f.page_content) for f in fragmentos]
        filas.append({
            "estrategia": nombre,
            "fragmentos": indice._collection.count(),
            "largo medio": round(np.mean(largos)),
            **evaluar(indice),
            "indexado (s)": round(segundos, 1),
        })
        indice.delete_collection()
    return pd.DataFrame(filas)


def experimento_recuperacion(paginas: list[Document], modelo) -> pd.DataFrame:
    """Compara formas de recuperar sobre el índice del proyecto (800/100, e5-small)."""
    indice, _ = indexar(chunking_recursivo(paginas, 800, 100), modelo)
    nombres = {
        "topk": f"top-{K} (similitud)",
        "topk8": "top-8 (similitud)",
        "umbral": f"top-{K} + umbral {config.UMBRAL_RELEVANCIA} (proyecto)",
        "mmr": f"MMR (k={K}, fetch_k=20, λ=0,5)",
        "filtro": f"top-{K} + filtro por paper",
    }
    filas = [{"recuperación": nombre, **evaluar(indice, modo)} for modo, nombre in nombres.items()]
    indice.delete_collection()
    return pd.DataFrame(filas)


def experimento_embeddings(paginas: list[Document]) -> pd.DataFrame:
    """Compara modelos de embeddings con el mismo chunking (800/100) y top-k."""
    fragmentos = chunking_recursivo(paginas, 800, 100)
    filas = []
    for nombre, (_, _, comentario) in MODELOS_EMBEDDINGS.items():
        inicio = time.time()
        modelo = crear_modelo(nombre)
        carga = time.time() - inicio
        indice, segundos = indexar(fragmentos, modelo)
        filas.append({
            "modelo": nombre,
            "comentario": comentario,
            "dimensión": len(modelo.embed_query("hola")),
            **evaluar(indice),
            **separacion(indice),
            "carga (s)": round(carga, 1),
            "indexado (s)": round(segundos, 1),
        })
        indice.delete_collection()
    return pd.DataFrame(filas)
