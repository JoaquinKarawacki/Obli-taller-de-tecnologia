"""Fase 4 — Agente (LangGraph).

Grafo del agente construido a mano (como la Parte 3/4 del Lab04, pero sin los
atajos de ``langgraph.prebuilt``):

    START -> agente --(¿pidió tools?)--> herramientas -> agente -> ... -> END
                     \\--(no)--> END

- ``agente``: el LLM con las tools disponibles. Decide si responde directo
  (small-talk, consultas generales) o si llama a una o varias tools (papers,
  dataset). Así se resuelve el tipo de consulta de forma dinámica, sin un nodo
  router aparte.
- ``herramientas``: ejecuta las tools que pidió el LLM y devuelve sus resultados
  como ``ToolMessage``; después el control vuelve al agente, que puede buscar
  de nuevo o generar la respuesta final.

Memoria (Fase 5, ver ``src/memoria.py``): el historial de cada conversación lo
guarda el checkpointer por ``thread_id``; los recuerdos del usuario (``user_id``)
se leen del store en cada turno y se agregan al prompt de sistema, y el propio
agente decide cuándo guardar uno nuevo con la tool ``guardar_recuerdo``.
Ambos IDs llegan en el config de cada invocación:
``{"configurable": {"thread_id": ..., "user_id": ...}}``.

IMPORTANTE: prohibido usar create_agent() / create_react_agent() ni abstracciones
de alto nivel que oculten la lógica del agente (ver PLAN.md). Por eso también el
nodo de tools y la condición de salida están escritos a mano (en vez de
``ToolNode`` / ``tools_condition``).
"""
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore

from src import config
from src.memoria import guardar_recuerdo, leer_recuerdos
from src.recuperacion import HERRAMIENTAS, SIN_EVIDENCIA

# Prompt de sistema: define el rol, cuándo usar cada tool y las reglas contra
# las alucinaciones. Se agrega al inicio de los mensajes en cada llamada al LLM
# (no se guarda en el estado, así no se repite en el historial).
PROMPT_SISTEMA = f"""Sos un asistente experto en ciberseguridad (phishing) y en los papers de la materia.
Tenés acceso a herramientas:
- buscar_en_papers / listar_papers: para preguntas sobre los documentos cargados (Attention, GPT, GPT-3, Scaling Laws, Subliminal Learning, Designing Data-Intensive Applications).
- buscar_en_dataset: para mostrar ejemplos reales de correos de phishing o legítimos, o comparar un correo sospechoso con casos conocidos.
- estadisticas_dataset: para preguntas de cantidades o composición del dataset.
- guardar_recuerdo: para recordar datos del usuario en futuras conversaciones.

Cómo decidir según el tipo de consulta:
1. Saludos, agradecimientos o charla (small-talk): respondé breve y amable, SIN usar herramientas.
2. Preguntas sobre los papers o el dataset: usá SIEMPRE la herramienta correspondiente antes de responder. Podés llamar varias herramientas o buscar de nuevo con otras palabras si el primer resultado no alcanza.
3. Preguntas generales de conocimiento (fuera de los papers y del dataset): podés responder con tu conocimiento general, pero SIEMPRE aclarando en la respuesta que no sale de los documentos cargados.

Reglas contra las alucinaciones:
- Para preguntas sobre los papers o el dataset, respondé SOLO con la información de los fragmentos recuperados. No inventes datos, cifras ni citas.
- Si la herramienta devuelve "{SIN_EVIDENCIA}" o los fragmentos no responden la pregunta, decí explícitamente que no tenés información suficiente en los documentos.
- Si una herramienta devuelve "ERROR", explicá que no pudiste consultar esa fuente.
- Citá la fuente de cada dato (título del paper y página, o corpus del correo).

Memoria a largo plazo:
- Si el usuario cuenta un dato duradero sobre sí mismo (nombre, trabajo, intereses, preferencias, nivel de conocimiento), guardalo con guardar_recuerdo: un dato por llamada y sin repetir lo que ya está en sus recuerdos.
- NO guardes preguntas puntuales, datos de los documentos ni datos sensibles (contraseñas, tarjetas, documentos de identidad).
- Usá los recuerdos para personalizar la respuesta (ej. llamarlo por su nombre o adaptar el nivel técnico).

Respondé de forma clara y concisa (salvo que el usuario pida más detalle), siempre en el mismo idioma en que escribe el usuario (español o inglés), aunque los documentos estén en inglés."""


def crear_llm() -> ChatGroq:
    """LLM de Groq. ``temperature=0`` para respuestas más estables y fieles al contexto.

    El plan gratuito de Groq limita los tokens de salida por minuto (1000 para este
    modelo). Por eso se apaga el "razonamiento" de qwen3 (gasta tokens de salida
    antes de cada respuesta) y, si igual se llega al límite, se reintenta esperando.
    """
    return ChatGroq(
        api_key=config.GROQ_API_KEY,
        model=config.MODELO_LLM,
        temperature=0,
        reasoning_effort="none",
        max_retries=6,
    )


def construir_agente(checkpointer, store):
    """Arma y compila el grafo de LangGraph del agente.

    Args:
        checkpointer: memoria a corto plazo (historial por ``thread_id``).
        store: memoria a largo plazo (recuerdos por ``user_id``).
    """
    herramientas = HERRAMIENTAS + [guardar_recuerdo]
    herramientas_por_nombre = {herramienta.name: herramienta for herramienta in herramientas}
    llm_con_tools = crear_llm().bind_tools(herramientas)

    def nodo_agente(estado: MessagesState, config: RunnableConfig, *, store: BaseStore):
        """El LLM decide: responder directo o pedir herramientas.

        Antes se agregan al prompt los recuerdos del usuario (memoria a largo plazo).
        """
        recuerdos = leer_recuerdos(store, config["configurable"]["user_id"])
        lista = "\n".join(f"- {recuerdo}" for recuerdo in recuerdos) or "(todavía no hay recuerdos)"
        prompt = f"{PROMPT_SISTEMA}\n\nRecuerdos guardados de este usuario:\n{lista}"
        mensajes = [SystemMessage(prompt)] + estado["messages"]
        return {"messages": [llm_con_tools.invoke(mensajes)]}

    def nodo_herramientas(estado: MessagesState, config: RunnableConfig, *, store: BaseStore):
        """Ejecuta cada tool que pidió el LLM y devuelve un ToolMessage por cada una."""
        resultados = []
        for llamada in estado["messages"][-1].tool_calls:
            herramienta = herramientas_por_nombre[llamada["name"]]
            argumentos = llamada["args"]
            if herramienta is guardar_recuerdo:
                # El usuario y el store salen de la sesión, no del LLM.
                argumentos = {**argumentos, "user_id": config["configurable"]["user_id"], "store": store}
            salida = herramienta.invoke(argumentos)
            resultados.append(ToolMessage(content=salida, tool_call_id=llamada["id"]))
        return {"messages": resultados}

    def decidir_siguiente(estado: MessagesState) -> str:
        """Si el último mensaje del LLM pide tools, se ejecutan; si no, termina."""
        if estado["messages"][-1].tool_calls:
            return "herramientas"
        return END

    grafo = StateGraph(MessagesState)
    grafo.add_node("agente", nodo_agente)
    grafo.add_node("herramientas", nodo_herramientas)
    grafo.add_edge(START, "agente")
    grafo.add_conditional_edges("agente", decidir_siguiente, ["herramientas", END])
    grafo.add_edge("herramientas", "agente")
    return grafo.compile(checkpointer=checkpointer, store=store)
