"""Ejecutable: prueba la memoria a corto y largo plazo y las sesiones en paralelo.

Guion:
  1. Ana se presenta en la conversación 1 y hace una pregunta  -> el agente guarda recuerdos.
  2. Ana abre la conversación 2                                -> recuerda quién es (largo plazo),
                                                                  pero no lo que habló en la 1.
  3. Se "reinicia" el programa y Ana retoma la conversación 1   -> sigue el contexto (corto plazo,
                                                                  persistido en SQLite).
  4. Ana y Beto escriben al mismo tiempo (dos hilos)           -> cada uno con su memoria.

Usa la memoria persistente (SQLite) con IDs nuevos en cada corrida, para no mezclar con pruebas anteriores.

Uso (desde la raíz del repo, con el venv):
    python -m scripts.probar_memoria
"""
import uuid
from concurrent.futures import ThreadPoolExecutor

from src import memoria
from src.agente import construir_agente

CORRIDA = uuid.uuid4().hex[:6]
ANA, BETO = f"ana-{CORRIDA}", f"beto-{CORRIDA}"


def crear_agente():
    """Agente con memoria persistente (como lo usará el chatbot)."""
    return construir_agente(memoria.crear_memoria_corto_plazo(), memoria.crear_memoria_largo_plazo())


def decir(agente, user_id: str, conversacion: str, texto: str) -> str:
    """Manda un mensaje en una conversación de un usuario y muestra la respuesta."""
    sesion = {"configurable": {"thread_id": f"{user_id}-{conversacion}", "user_id": user_id}}
    resultado = agente.invoke({"messages": [("user", texto)]}, sesion)
    respuesta = resultado["messages"][-1].content
    print(f"\n[{user_id} / conv {conversacion}] {texto}\n  -> {respuesta}")
    return respuesta


if __name__ == "__main__":
    agente = crear_agente()

    print("\n### 1. Conversación 1: Ana se presenta")
    decir(agente, ANA, "1", "Hola! Soy Ana, trabajo en el área de seguridad informática de un banco.")
    decir(agente, ANA, "1", "En una frase: ¿qué es el multi-head attention?")
    print(f"\nRecuerdos de Ana: {memoria.leer_recuerdos(agente.store, ANA)}")

    print("\n### 2. Conversación 2 (nueva): largo plazo sí, corto plazo de la 1 no")
    decir(agente, ANA, "2", "¿Te acordás cómo me llamo y a qué me dedico?")
    decir(agente, ANA, "2", "¿Qué te pregunté en nuestra otra conversación?")

    print("\n### 3. 'Reinicio' del programa y Ana retoma la conversación 1")
    agente = crear_agente()  # conexiones nuevas a los SQLite, nada en RAM
    decir(agente, ANA, "1", "¿Qué te había preguntado recién?")

    print("\n### 4. Ana y Beto en paralelo")
    with ThreadPoolExecutor() as hilos:
        hilos.submit(decir, agente, BETO, "1", "Hola, ¿sabés cómo me llamo?")
        hilos.submit(decir, agente, ANA, "3", "Hola, ¿sabés cómo me llamo?")
    print(f"\nRecuerdos de Beto: {memoria.leer_recuerdos(agente.store, BETO)}")
