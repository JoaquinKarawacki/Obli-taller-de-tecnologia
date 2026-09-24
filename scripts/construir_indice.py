"""Ejecutable: construye el índice vectorial del dataset y lo prueba.

Uso (desde la raíz del repo, con el venv):
    python -m scripts.construir_indice
"""
from src.ingesta import construir_indice_dataset

# Consultas de prueba EN ESPAÑOL sobre correos EN INGLÉS (para validar el
# retrieval cross-lingual que dan los embeddings multilingües).
CONSULTAS_PRUEBA = [
    "correo que pide datos bancarios o contraseña",
    "reunión de trabajo",
]

if __name__ == "__main__":
    indice = construir_indice_dataset()

    print("\n=== Pruebas de búsqueda semántica (consulta en español) ===")
    for consulta in CONSULTAS_PRUEBA:
        print(f"\nConsulta: {consulta!r}")
        resultados = indice.similarity_search(consulta, k=3)
        for i, doc in enumerate(resultados, 1):
            etiqueta = "PHISHING" if doc.metadata.get("label") == 1 else "legítimo"
            extracto = doc.page_content[:120].replace("\n", " ")
            print(f"  {i}. [{etiqueta}] fuente={doc.metadata.get('fuente')} | {extracto}")
