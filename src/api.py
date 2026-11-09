"""Opcional — API REST con FastAPI.

Expone el mismo agente que usa el chatbot por consola (``src/chatbot.py``) para que
otra aplicación (una web, un bot, otro servicio) lo pueda usar por HTTP. No duplica
lógica: reutiliza ``construir_agente`` y las dos memorias persistentes.

Endpoints:

    GET  /salud                                          estado del servicio
    POST /chat                                           enviar un mensaje al agente
    GET  /usuarios/{user_id}/conversaciones              listar conversaciones
    GET  /usuarios/{user_id}/conversaciones/{thread_id}  historial de una conversación
    GET  /usuarios/{user_id}/recuerdos                   memoria a largo plazo

Los mismos identificadores que en el chatbot:
  - ``user_id``: el usuario (memoria a largo plazo).
  - ``thread_id``: la conversación (memoria a corto plazo). Si ``/chat`` no lo
    recibe, empieza una conversación nueva y devuelve su id para continuarla.

Igual que ``/retomar`` en el chatbot, un usuario solo puede continuar o leer sus
propias conversaciones: si el ``thread_id`` no está en su índice, la API responde 404.

Uso (desde la raíz del repo, con el entorno virtual activado):
    uvicorn src.api:app
La documentación interactiva (Swagger) queda en http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from groq import RateLimitError
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from src import config
from src.agente import construir_agente
from src.chatbot import _herramientas_usadas, _nuevo_thread_id
from src.memoria import (
    crear_memoria_corto_plazo,
    crear_memoria_largo_plazo,
    leer_recuerdos,
    listar_conversaciones,
    registrar_conversacion,
)


# --- Modelos de entrada y salida (validados por Pydantic) ---


class MensajeEntrada(BaseModel):
    user_id: str = Field(min_length=1, max_length=64, examples=["ana"])
    mensaje: str = Field(min_length=1, max_length=4000, examples=["¿Qué es la atención multi-cabeza?"])
    thread_id: str | None = Field(default=None, description="Vacío para empezar una conversación nueva.")


class RespuestaChat(BaseModel):
    thread_id: str
    respuesta: str
    herramientas: list[str] = Field(description="Tools que usó el agente en este turno.")


class Conversacion(BaseModel):
    thread_id: str
    titulo: str


class MensajeHistorial(BaseModel):
    rol: str  # "usuario" o "asistente"
    contenido: str


# --- Aplicación ---


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    """Construye el agente UNA sola vez al arrancar (carga embeddings y abre las memorias)."""
    if not config.GROQ_API_KEY:
        raise RuntimeError("Falta GROQ_API_KEY en .env (ver README).")
    app.state.store = crear_memoria_largo_plazo()
    app.state.agente = construir_agente(crear_memoria_corto_plazo(), app.state.store)
    yield


app = FastAPI(
    title="Agentic RAG — Taller de Tecnologías 2",
    description="Chatbot sobre papers y un dataset de phishing, con memoria a corto y largo plazo.",
    version="1.0",
    lifespan=ciclo_de_vida,
)


def _verificar_conversacion(user_id: str, thread_id: str) -> None:
    """404 si la conversación no pertenece al usuario (no se puede leer la de otro)."""
    if thread_id not in listar_conversaciones(app.state.store, user_id):
        raise HTTPException(404, f"El usuario '{user_id}' no tiene una conversación con id '{thread_id}'.")


# Los endpoints son funciones comunes (def, no async def): FastAPI las ejecuta en un
# pool de hilos, así una consulta lenta al LLM no bloquea a los demás usuarios. Es
# el mismo esquema de sesiones en paralelo que se probó en la sección 9 del notebook.


@app.get("/salud")
def salud() -> dict:
    """Indica que el servicio está levantado y qué modelos usa."""
    return {"estado": "ok", "llm": config.MODELO_LLM, "embeddings": config.MODELO_EMBEDDINGS}


@app.post("/chat", response_model=RespuestaChat)
def chat(entrada: MensajeEntrada) -> RespuestaChat:
    """Envía un mensaje al agente y devuelve su respuesta.

    Sin ``thread_id`` empieza una conversación nueva; con el ``thread_id`` que
    devolvió una llamada anterior, la continúa con todo su contexto.
    """
    if entrada.thread_id:
        _verificar_conversacion(entrada.user_id, entrada.thread_id)
        thread_id = entrada.thread_id
    else:
        thread_id = _nuevo_thread_id()

    sesion = {"configurable": {"thread_id": thread_id, "user_id": entrada.user_id}}
    registrar_conversacion(app.state.store, entrada.user_id, thread_id, entrada.mensaje)
    try:
        resultado = app.state.agente.invoke({"messages": [("user", entrada.mensaje)]}, sesion)
    except RateLimitError:
        raise HTTPException(429, "Se agotó la cuota de tokens de Groq. Probá de nuevo más tarde.")
    except Exception as error:
        raise HTTPException(502, f"Error al consultar al agente: {error}")

    mensajes = resultado["messages"]
    return RespuestaChat(
        thread_id=thread_id,
        respuesta=mensajes[-1].content,
        herramientas=_herramientas_usadas(mensajes),
    )


@app.get("/usuarios/{user_id}/conversaciones", response_model=list[Conversacion])
def conversaciones(user_id: str) -> list[Conversacion]:
    """Lista las conversaciones guardadas del usuario, de la más vieja a la más nueva."""
    return [
        Conversacion(thread_id=thread_id, titulo=titulo)
        for thread_id, titulo in listar_conversaciones(app.state.store, user_id).items()
    ]


@app.get("/usuarios/{user_id}/conversaciones/{thread_id}", response_model=list[MensajeHistorial])
def historial(user_id: str, thread_id: str) -> list[MensajeHistorial]:
    """Devuelve las preguntas y respuestas de una conversación, leídas del checkpointer.

    No incluye las llamadas a tools ni los fragmentos recuperados.
    """
    _verificar_conversacion(user_id, thread_id)
    estado = app.state.agente.get_state({"configurable": {"thread_id": thread_id, "user_id": user_id}})
    salida = []
    for mensaje in estado.values.get("messages", []):
        if isinstance(mensaje, HumanMessage):
            salida.append(MensajeHistorial(rol="usuario", contenido=mensaje.content))
        elif isinstance(mensaje, AIMessage) and not mensaje.tool_calls:
            salida.append(MensajeHistorial(rol="asistente", contenido=mensaje.content))
    return salida


@app.get("/usuarios/{user_id}/recuerdos", response_model=list[str])
def recuerdos(user_id: str) -> list[str]:
    """Devuelve lo que el agente recuerda del usuario (memoria a largo plazo)."""
    return leer_recuerdos(app.state.store, user_id)
