"""Fase 6 — Chatbot.

Loop de conversación por consola que usa el agente (Fase 4) con las dos memorias
persistentes (Fase 5). Es la interfaz que vuelve usable todo el sistema.

Cómo se "encadena" la conversación: en cada turno se le manda al agente SOLO el
mensaje nuevo. El historial no lo guarda este loop, lo guarda el checkpointer de
LangGraph indexado por ``thread_id``: antes de cada paso carga los mensajes previos
de esa conversación y después guarda el estado actualizado. Por eso alcanza con
mandar el mensaje nuevo y cambiar el ``thread_id`` para empezar otra conversación.

Identificadores (vienen en el ``config`` de cada invocación):
  - ``thread_id``: la conversación (memoria a corto plazo). Uno nuevo = empezar de cero.
  - ``user_id``:   el usuario (memoria a largo plazo, recuerdos compartidos entre sus
                   conversaciones).

El chatbot es multilingüe: se puede preguntar en español o inglés.
"""
import uuid

from src.agente import construir_agente
from src.memoria import crear_memoria_corto_plazo, crear_memoria_largo_plazo

# Comandos del chat (empiezan con "/"). El resto se trata como mensaje para el agente.
AYUDA = """\
Comandos disponibles:
  /nueva          -> empieza una conversación nueva (se olvida el contexto del chat actual)
  /usuario <id>   -> cambia de usuario (otros recuerdos de largo plazo) y empieza conversación nueva
  /ayuda          -> muestra esta ayuda
  /salir          -> termina el programa

Podés preguntar en español o inglés sobre:
  - los papers cargados (Attention, GPT, GPT-3, Scaling Laws, Subliminal Learning,
    Designing Data-Intensive Applications),
  - el dataset de correos de phishing/legítimos (ejemplos, patrones, estadísticas).
Si contás algo sobre vos (nombre, trabajo, intereses), el asistente puede recordarlo."""


def _nuevo_thread_id() -> str:
    """Genera un id de conversación nuevo (así el checkpointer arranca de cero)."""
    return uuid.uuid4().hex[:8]


def _herramientas_usadas(mensajes) -> list[str]:
    """Nombres de las tools que el agente pidió en este turno (para evidenciar el RAG)."""
    return [
        llamada["name"]
        for mensaje in mensajes
        for llamada in getattr(mensaje, "tool_calls", [])
    ]


def iniciar_chat(user_id_inicial: str = "usuario") -> None:
    """Arranca el loop de chat por consola.

    Construye el agente UNA sola vez (carga el modelo de embeddings y abre las bases
    de memoria) y después entra en el bucle de conversación.
    """
    print("Cargando modelo y memorias (puede tardar unos segundos la primera vez)...")
    agente = construir_agente(crear_memoria_corto_plazo(), crear_memoria_largo_plazo())

    user_id = user_id_inicial
    thread_id = _nuevo_thread_id()

    print("\n=== Agentic RAG — Chatbot (Taller de Tecnologías 2) ===")
    print(AYUDA)

    while True:
        try:
            texto = input(f"\n[{user_id} · conv {thread_id}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n¡Hasta luego!")
            break

        if not texto:
            continue

        # --- Comandos ---
        if texto.startswith("/"):
            partes = texto.split(maxsplit=1)
            comando = partes[0].lower()

            if comando == "/salir":
                print("¡Hasta luego!")
                break
            if comando == "/ayuda":
                print(AYUDA)
                continue
            if comando == "/nueva":
                thread_id = _nuevo_thread_id()
                print(f"(nueva conversación: {thread_id})")
                continue
            if comando == "/usuario":
                if len(partes) < 2 or not partes[1].strip():
                    print("Uso: /usuario <id>  (por ejemplo: /usuario ana)")
                    continue
                user_id = partes[1].strip()
                thread_id = _nuevo_thread_id()
                print(f"(ahora sos '{user_id}', conversación nueva: {thread_id})")
                continue

            print(f"Comando desconocido: {comando}. Escribí /ayuda para ver los comandos.")
            continue

        # --- Mensaje normal: se lo mandamos al agente ---
        sesion = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        try:
            resultado = agente.invoke({"messages": [("user", texto)]}, sesion)
        except Exception as error:
            # No cortamos la sesión por un error de la API (ej. rate limit de Groq).
            print(f"[error al consultar al agente: {error}]")
            continue

        herramientas = _herramientas_usadas(resultado["messages"])
        print(f"\n{resultado['messages'][-1].content}")
        if herramientas:
            print(f"  (usó: {', '.join(herramientas)})")
