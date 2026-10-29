"""Fase 5 — Memoria y sesiones.

- Memoria a corto plazo por ``thread_id`` (checkpointer): LangGraph guarda el estado
  (los mensajes) de cada conversación después de cada paso. Con el mismo
  ``thread_id`` la conversación se retoma donde quedó; con otro, empieza de cero.
- Memoria a largo plazo por ``user_id`` (store): datos del usuario que valen para
  todas sus conversaciones. El agente decide qué guardar (tool ``guardar_recuerdo``)
  y en cada turno se le muestran los recuerdos del usuario para que los use.
- Sesiones de usuario en paralelo: cada invocación lleva su ``user_id`` y su
  ``thread_id`` en el ``config``, así que conversaciones de distintos usuarios no
  se mezclan.

Ambas memorias se persisten en SQLite (``memoria.sqlite3``): se puede cerrar el
programa y retomar una conversación después, sin perder el contexto.
"""
import sqlite3
import uuid
from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.base import BaseStore
from langgraph.store.sqlite import SqliteStore

from src import config


def _conectar() -> sqlite3.Connection:
    # check_same_thread=False: varias sesiones en paralelo (hilos) usan la misma conexión.
    # isolation_level=None (autocommit): el checkpointer y el store manejan sus propias
    # transacciones; sin esto SQLite deja transacciones abiertas y da "database is locked".
    return sqlite3.connect(config.ARCHIVO_MEMORIA, check_same_thread=False, isolation_level=None)


def crear_memoria_corto_plazo() -> SqliteSaver:
    """Checkpointer: guarda el historial de cada conversación (``thread_id``)."""
    return SqliteSaver(_conectar())


def crear_memoria_largo_plazo() -> SqliteStore:
    """Store: guarda los recuerdos de cada usuario (``user_id``), compartidos entre conversaciones."""
    store = SqliteStore(_conectar())
    store.setup()  # crea las tablas la primera vez
    return store


def _espacio(user_id: str) -> tuple[str, ...]:
    """Namespace del store donde viven los recuerdos de un usuario."""
    return ("usuarios", user_id, "recuerdos")


def leer_recuerdos(store: BaseStore, user_id: str) -> list[str]:
    """Devuelve todos los recuerdos guardados de un usuario (son pocos: no hace falta buscar)."""
    return [item.value["texto"] for item in store.search(_espacio(user_id), limit=100)]


@tool
def guardar_recuerdo(
    recuerdo: str,
    user_id: Annotated[str, InjectedToolArg],
    store: Annotated[BaseStore, InjectedToolArg],
) -> str:
    """Guarda en la memoria a largo plazo un dato duradero sobre el usuario
    (nombre, trabajo, intereses, preferencias, nivel de conocimiento), para
    recordarlo en futuras conversaciones.

    Args:
        recuerdo: el dato en una frase corta, ej. "Se llama Ana y trabaja en un banco".
    """
    # user_id y store no los completa el LLM (no los ve): los agrega el nodo de
    # herramientas con los datos de la sesión, así un usuario no puede escribir en
    # la memoria de otro.
    store.put(_espacio(user_id), str(uuid.uuid4()), {"texto": recuerdo})
    return "Recuerdo guardado."
