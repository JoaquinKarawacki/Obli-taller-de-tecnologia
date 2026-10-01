"""Ejecutable: prueba la recuperación y ayuda a calibrar UMBRAL_RELEVANCIA.

Compara el mejor puntaje de consultas PERTINENTES (la respuesta está en los
documentos) con el de consultas NO PERTINENTES (nada que ver). El umbral debe
quedar entre ambos grupos: por encima de las no pertinentes y por debajo de las
pertinentes.

Uso (desde la raíz del repo, con el venv):
    python -m scripts.probar_recuperacion
"""
from src import config
from src.recuperacion import buscar_en_papers, buscar_fragmentos

PERTINENTES = [
    "¿qué es el mecanismo de atención multi-cabeza?",
    "¿cuántos parámetros tiene GPT-3?",
    "¿cómo escala la pérdida con el tamaño del modelo?",
    "¿qué es el aprendizaje subliminal en modelos de lenguaje?",
    "replicación líder-seguidor en bases de datos",
    "What is the pre-training objective used in GPT?",
]
NO_PERTINENTES = [
    "receta de lasaña vegetariana",
    "¿quién ganó el mundial de fútbol 2022?",
    "¿cuál es la capital de Australia?",
    "hola, ¿cómo estás?",
    "mejores playas de Uruguay para el verano",
]


def mejor_puntaje(consulta: str) -> tuple[float, str]:
    """Devuelve el puntaje y la fuente del fragmento más relevante (sin umbral)."""
    umbral = config.UMBRAL_RELEVANCIA
    config.UMBRAL_RELEVANCIA = -1.0  # se desactiva el umbral para ver todos los puntajes
    try:
        resultados = buscar_fragmentos(consulta, config.COLECCION_PAPERS, k=1)
    finally:
        config.UMBRAL_RELEVANCIA = umbral
    doc, puntaje = resultados[0]
    return puntaje, f"{doc.metadata['titulo']} pág. {doc.metadata['pagina']}"


if __name__ == "__main__":
    for nombre, consultas in (("PERTINENTES", PERTINENTES), ("NO PERTINENTES", NO_PERTINENTES)):
        print(f"\n=== {nombre} ===")
        for consulta in consultas:
            puntaje, fuente = mejor_puntaje(consulta)
            print(f"  {puntaje:.3f}  {consulta!r}  ->  {fuente}")

    print(f"\nUmbral actual: {config.UMBRAL_RELEVANCIA}")
    print("\n=== Ejemplo de salida de la tool (lo que ve el LLM) ===")
    print(buscar_en_papers.invoke({"consulta": "multi-head attention", "titulo_paper": "attention", "cantidad": 2}))
    print("\n=== Consulta no pertinente ===")
    print(buscar_en_papers.invoke({"consulta": "receta de lasaña vegetariana"}))
