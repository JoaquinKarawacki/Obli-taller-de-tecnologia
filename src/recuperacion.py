"""Fase 3 — Recuperación (RAG).

Búsqueda semántica top-k sobre las colecciones de Chroma, expuesta como
herramientas (tools) para que el agente decida cuándo y dónde buscar.

Es el mismo pipeline del Demo_vectoriales_crawler (``search`` + ``format_message``)
con dos agregados:
  - un UMBRAL de relevancia: si ningún fragmento es suficientemente parecido a la
    consulta, se devuelve "SIN EVIDENCIA" en vez de fragmentos irrelevantes. Así el
    agente sabe que debe admitir que no tiene información (control de alucinaciones).
  - metadatos para citar la fuente (paper + página, o corpus + etiqueta del correo).
"""
import difflib
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool

from src import config
from src import datos
from src import ingesta

SIN_EVIDENCIA = "SIN EVIDENCIA"


@lru_cache(maxsize=None)
def _indice(coleccion: str) -> Chroma:
    """Abre la colección una sola vez y la reutiliza en todas las búsquedas."""
    return ingesta.cargar_indice(coleccion)


def buscar_fragmentos(
    consulta: str,
    coleccion: str,
    k: int = config.K_FRAGMENTOS,
    filtro: dict | None = None,
) -> list[tuple[Document, float]]:
    """Devuelve hasta ``k`` fragmentos relevantes con su puntaje (0 a 1).

    Se descartan los fragmentos con relevancia menor a ``UMBRAL_RELEVANCIA``.

    Args:
        consulta: texto de la búsqueda (puede estar en español o inglés).
        coleccion: nombre de la colección de Chroma (dataset o papers).
        k: cantidad de fragmentos a recuperar.
        filtro: filtro de metadatos de Chroma, ej. ``{"titulo": "..."}``.
    """
    indice = _indice(coleccion)
    if indice._collection.count() == 0:
        # Sin esto, una colección no construida parecería "no hay nada relevante".
        raise RuntimeError(f"La colección '{coleccion}' está vacía: falta construir el índice (ver README).")
    resultados = indice.similarity_search_with_relevance_scores(
        consulta, k=k, filter=filtro
    )
    return [(doc, puntaje) for doc, puntaje in resultados if puntaje >= config.UMBRAL_RELEVANCIA]


def formatear_fragmentos(resultados: list[tuple[Document, float]]) -> str:
    """Arma el bloque de contexto que se le pasa al LLM (como ``format_message``).

    Cada fragmento va numerado y con su fuente, para que la respuesta pueda citarla.
    """
    if not resultados:
        return f"{SIN_EVIDENCIA}: no se encontraron fragmentos relevantes para esta consulta."

    bloques = []
    for i, (doc, puntaje) in enumerate(resultados, 1):
        meta = doc.metadata
        if "titulo" in meta:  # fragmento de un paper
            fuente = f"{meta['titulo']}, pág. {meta['pagina']}"
        else:                 # fragmento de un correo del dataset
            etiqueta = "PHISHING" if meta.get("label") == 1 else "LEGÍTIMO"
            fuente = f"correo {etiqueta} | corpus {meta.get('fuente')}"
            if meta.get("sender"):
                fuente += f" | remitente {meta['sender']}"
        bloques.append(f"[{i}] Fuente: {fuente} (relevancia {puntaje:.2f})\n{doc.page_content}")
    return "\n\n".join(bloques)


def _resolver_titulo(texto: str) -> str | None:
    """Encuentra el título exacto de un paper a partir de un nombre aproximado.

    El LLM puede escribir "attention", "el paper de GPT-3" o "Kleppmann": se busca
    primero por coincidencia parcial y si no, por similitud de texto.
    """
    titulos = list(config.TITULOS_PAPERS.values())
    texto = texto.lower().strip()
    for titulo in titulos:
        if texto in titulo.lower():
            return titulo
    parecidos = difflib.get_close_matches(texto, [t.lower() for t in titulos], n=1, cutoff=0.4)
    if parecidos:
        return next(t for t in titulos if t.lower() == parecidos[0])
    return None


# ---------------------------------------------------------------------------
# Tools para el agente
# ---------------------------------------------------------------------------
# Los docstrings son lo que el LLM lee para decidir qué herramienta usar, por eso
# explican claramente cuándo conviene cada una.


@tool
def buscar_en_papers(consulta: str, titulo_paper: str = "", cantidad: int = config.K_FRAGMENTOS) -> str:
    """Busca fragmentos en los papers/documentos cargados (Transformers/Attention,
    GPT, GPT-3, Scaling Laws, Subliminal Learning y el libro Designing
    Data-Intensive Applications).

    Args:
        consulta: qué buscar, en español o inglés.
        titulo_paper: opcional; nombre (aunque sea aproximado) del paper para
            buscar solo dentro de él.
        cantidad: fragmentos a recuperar (4 por defecto; usar hasta 10 para
            resúmenes o preguntas amplias).
    """
    filtro = None
    if titulo_paper:
        titulo = _resolver_titulo(titulo_paper)
        if titulo is None:
            return f"No hay ningún documento cargado que coincida con '{titulo_paper}'. Documentos disponibles: {listar_papers.invoke({})}"
        filtro = {"titulo": titulo}
    cantidad = max(1, min(int(cantidad), 10))
    try:
        return formatear_fragmentos(buscar_fragmentos(consulta, config.COLECCION_PAPERS, cantidad, filtro))
    except RuntimeError as error:
        return f"ERROR: {error}"


@tool
def buscar_en_dataset(consulta: str, tipo_correo: str = "") -> str:
    """Busca correos parecidos en el dataset de correos de phishing y legítimos.
    Sirve para mostrar ejemplos reales y patrones de phishing, o comparar un
    correo sospechoso con casos conocidos.

    Args:
        consulta: qué buscar (tema, texto de un correo, patrón), en español o inglés.
        tipo_correo: opcional; "phishing" o "legitimo" para buscar solo ese tipo.
    """
    filtro = None
    if tipo_correo.lower().startswith("phish"):
        filtro = {"label": 1}
    elif tipo_correo.lower().startswith("leg"):
        filtro = {"label": 0}
    try:
        return formatear_fragmentos(buscar_fragmentos(consulta, config.COLECCION_DATASET, filtro=filtro))
    except RuntimeError as error:
        return f"ERROR: {error}"


@tool
def listar_papers() -> str:
    """Devuelve la lista de documentos/papers disponibles para consultar."""
    return "\n".join(f"- {titulo}" for titulo in config.TITULOS_PAPERS.values())


@lru_cache(maxsize=1)
def _resumen_dataset() -> str:
    """Calcula una sola vez las estadísticas del dataset consolidado."""
    df = datos.cargar_consolidado()
    por_label = df["label"].value_counts().to_dict()
    por_fuente = df.groupby("fuente")["label"].agg(["count", "sum"])
    lineas = [
        f"Total de correos: {len(df)}",
        f"Phishing: {por_label.get(1, 0)} | Legítimos: {por_label.get(0, 0)}",
        f"Correos indexados para búsqueda semántica: muestra balanceada de {config.TAMANO_MUESTRA}",
        "Por corpus de origen (total / phishing):",
    ]
    for fuente, fila in por_fuente.iterrows():
        lineas.append(f"  - {fuente}: {fila['count']} / {fila['sum']}")
    return "\n".join(lineas)


@tool
def estadisticas_dataset() -> str:
    """Devuelve estadísticas del dataset de correos: total, cantidad de phishing y
    legítimos, y desglose por corpus de origen. Usar para preguntas de conteo o
    sobre la composición del dataset (la búsqueda semántica no sirve para contar).
    """
    try:
        return _resumen_dataset()
    except FileNotFoundError:
        return "No se encontró el dataset consolidado en data/ (ver README)."


HERRAMIENTAS = [buscar_en_papers, buscar_en_dataset, listar_papers, estadisticas_dataset]
