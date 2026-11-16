"""Ejecutable: prueba el agente con consultas de distinto tipo.

Muestra, para cada consulta, qué herramientas decidió usar el agente y la
respuesta final. Sirve para verificar que decide bien según el tipo de consulta
(papers / dataset / general / small-talk) y que admite cuando no sabe.

Uso (desde la raíz del repo, con el venv):
    python -m scripts.probar_agente
"""
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from src.agente import construir_agente

CONSULTAS_PRUEBA = [
    "Hola! cómo andás?",                                         # small-talk
    "¿Qué es el multi-head attention según el paper de Attention?",  # papers (ES)
    "What does GPT-3 say about few-shot learning?",                # papers (EN)
    "¿Cuántos correos de phishing hay en el dataset?",           # estadísticas
    "Mostrame un ejemplo de correo de phishing que pida la contraseña del banco",  # dataset
    "¿Cuál es la capital de Francia?",                           # general
    "¿Qué dice el paper de Attention sobre el mundial de fútbol 2022?",  # sin evidencia
]


def herramientas_usadas(mensajes) -> list[str]:
    """Lista las tools que pidió el LLM durante la ejecución (con sus argumentos)."""
    return [
        f"{llamada['name']}({llamada['args']})"
        for mensaje in mensajes
        for llamada in getattr(mensaje, "tool_calls", [])
    ]


if __name__ == "__main__":
    # Memoria en RAM (no toca memoria.sqlite3) y una conversación nueva por
    # consulta: cada pregunta se prueba sola, sin contexto de las anteriores.
    agente = construir_agente(InMemorySaver(), InMemoryStore())
    for i, consulta in enumerate(CONSULTAS_PRUEBA):
        sesion = {"configurable": {"thread_id": f"prueba-{i}", "user_id": "prueba"}}
        resultado = agente.invoke({"messages": [("user", consulta)]}, sesion)
        print(f"\n=== {consulta}")
        print(f"Herramientas: {herramientas_usadas(resultado['messages']) or 'ninguna'}")
        print(f"Respuesta: {resultado['messages'][-1].content}")
