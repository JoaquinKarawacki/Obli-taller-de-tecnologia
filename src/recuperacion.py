"""Fase 3 — Recuperación (RAG).

Búsqueda semántica top-k sobre las colecciones de Chroma. Se expone como
herramienta (tool) para que el agente la invoque cuando lo necesite.
"""
# from src import config


def buscar_fragmentos(consulta: str, coleccion: str, k: int = 4):
    """Devuelve los ``k`` fragmentos más relevantes para la consulta.

    Args:
        consulta: texto de la búsqueda (puede estar en español o inglés).
        coleccion: nombre de la colección de Chroma (dataset o papers).
        k: cantidad de fragmentos a recuperar.
    """
    # TODO Fase 3: búsqueda semántica en Chroma
    raise NotImplementedError
