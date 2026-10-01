"""Ejecutable: construye el índice vectorial de los papers y lo prueba.

Antes, copiar los PDFs de la materia a ``data/papers/``.

Uso (desde la raíz del repo, con el venv):
    python -m scripts.construir_indice_papers
"""
from src.ingesta import construir_indice_papers

# Consultas de prueba en español sobre papers en inglés (retrieval cross-lingual).
CONSULTAS_PRUEBA = [
    "¿qué es el mecanismo de atención multi-cabeza?",
    "¿cómo escala la pérdida con el tamaño del modelo?",
    "replicación líder-seguidor en bases de datos",
]

if __name__ == "__main__":
    indice = construir_indice_papers()

    print("\n=== Pruebas de búsqueda semántica (consulta en español) ===")
    for consulta in CONSULTAS_PRUEBA:
        print(f"\nConsulta: {consulta!r}")
        resultados = indice.similarity_search_with_relevance_scores(consulta, k=3)
        for i, (doc, puntaje) in enumerate(resultados, 1):
            extracto = doc.page_content[:100]
            print(f"  {i}. ({puntaje:.2f}) {doc.metadata['titulo']} pág. {doc.metadata['pagina']} | {extracto}")
