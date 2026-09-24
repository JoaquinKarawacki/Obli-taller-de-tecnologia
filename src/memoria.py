"""Fase 5 — Memoria y sesiones.

- Memoria a corto plazo por ``thread_id`` (checkpointer): permite pausar y retomar
  conversaciones sin perder contexto.
- Memoria a largo plazo por ``user_id`` (store): información del usuario compartida
  entre conversaciones.
- Sesiones de usuario en paralelo, identificadas de forma única.
"""
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.store.memory import InMemoryStore


def crear_memoria_corto_plazo():
    """Devuelve el checkpointer para la memoria a corto plazo."""
    # TODO Fase 5: MemorySaver (luego SqliteSaver para persistir)
    raise NotImplementedError


def crear_memoria_largo_plazo():
    """Devuelve el store para la memoria a largo plazo."""
    # TODO Fase 5: InMemoryStore (decidir qué guardar y cuándo usarlo)
    raise NotImplementedError
