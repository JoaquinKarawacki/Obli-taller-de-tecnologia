"""Opcional — comparación de LLMs.

Cubre el opcional "Evaluar diferentes LLMs y comparar la calidad de las respuestas".
Ejecuta el MISMO agente (mismo grafo, prompt, tools y memoria) cambiando solo el
LLM, sobre una batería de casos tomada de las pruebas del notebook (sección 8).

Para cada caso se mide:
  - ``tools ok``: si usó exactamente las tools esperadas (mismo criterio que la
    sección 8 del notebook).
  - ``dato ok``: si la respuesta contiene el dato verificable del caso (por ejemplo
    "175" para los parámetros de GPT-3, o que admita no tener información). Es un
    control automático y parcial: la fidelidad completa se revisa leyendo las
    respuestas, que quedan guardadas.
  - tiempo, llamadas al LLM y tokens de entrada y salida.

Los LLMs se ejecutan en paralelo, uno por hilo: Groq cuenta la cuota por modelo,
así que no compiten entre sí. Los casos de cada modelo van en orden, porque el de
seguimiento depende del anterior.

Los resultados se guardan en ``resultados/comparacion_llms.json`` para mostrarlos en
el notebook sin volver a gastar cuota. Para regenerarlos (desde la raíz del repo):
    python -m src.comparacion_llms
"""
import json
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd
from langchain_core.callbacks import BaseCallbackHandler

from src import config

MODELOS = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
ARCHIVO_RESULTADOS = config.RAIZ / "resultados" / "comparacion_llms.json"

# (tipo, pregunta, tools esperadas, conversación, patrón del dato verificable o None)
# Los casos "contexto" y "seguimiento" comparten conversación: el segundo depende del primero.
CASOS = [
    ("small-talk", "Hola! cómo andás?", set(), "c1", None),
    ("general", "¿Cuál es la capital de Francia?", set(), "c2", r"par[ií]s"),
    ("papers ES", "¿Cuántos parámetros tiene GPT-3?", {"buscar_en_papers"}, "c3", r"175"),
    ("papers EN", "Which optimizer did the Transformer use, and with which beta values?",
     {"buscar_en_papers"}, "c4", r"0[.,]98"),
    ("sin evidencia", "¿Qué dice el paper de Attention sobre el mundial de fútbol 2022?",
     {"buscar_en_papers"}, "c5", r"no (tengo|hay|encontr|menciona|contiene|trata|aparece)"),
    ("estadística", "¿Cuántos correos de phishing hay en el dataset?", {"estadisticas_dataset"}, "c6", r"42[.,\s\u00a0\u202f]?891"),
    ("dataset", "Mostrame un ejemplo de correo de phishing bancario", {"buscar_en_dataset"}, "c7", None),
    ("contexto", "¿Qué es multi-head attention?", {"buscar_en_papers"}, "c8", None),
    ("seguimiento", "¿Y cuántas cabezas usa el modelo big?", {"buscar_en_papers"}, "c8", r"\b16\b"),
    ("memoria", "Me llamo Ana y trabajo en un banco, recordalo", {"guardar_recuerdo"}, "c9", None),
]


def verificar_dato(tipo: str, respuesta: str, error: str = "") -> bool | None:
    """True/False si la respuesta contiene el dato verificable del caso; None si no tiene."""
    patron = next(caso[4] for caso in CASOS if caso[0] == tipo)
    if not patron or error:
        return None
    return bool(re.search(patron, respuesta, re.IGNORECASE))


class ContadorLLM(BaseCallbackHandler):
    """Cuenta las llamadas al LLM y los tokens de un turno (incluye reintentos y el
    control de recuerdos, que no quedan guardados en el historial)."""

    def __init__(self):
        self.llamadas = self.entrada = self.salida = 0

    def on_llm_end(self, response, **kwargs):
        self.llamadas += 1
        for generaciones in response.generations:
            for generacion in generaciones:
                uso = getattr(getattr(generacion, "message", None), "usage_metadata", None)
                if uso:
                    self.entrada += uso.get("input_tokens", 0)
                    self.salida += uso.get("output_tokens", 0)


def _usuario(modelo: str) -> str:
    """user_id de cada modelo. El store de LangGraph no admite puntos en el namespace."""
    return "comparacion-" + re.sub(r"[^a-z0-9-]", "-", modelo.lower())


def evaluar_modelo(modelo: str, agente, store) -> list[dict]:
    """Ejecuta toda la batería con un LLM y devuelve una fila por caso."""
    from src.chatbot import _herramientas_usadas
    from src.memoria import leer_recuerdos

    usuario = _usuario(modelo)
    filas = []
    for tipo, pregunta, esperadas, conversacion, patron in CASOS:
        contador = ContadorLLM()
        sesion = {"configurable": {"thread_id": f"{modelo}-{conversacion}", "user_id": usuario}, "callbacks": [contador]}
        inicio = time.time()
        try:
            mensajes = agente.invoke({"messages": [("user", pregunta)]}, sesion)["messages"]
            respuesta, usadas, error = mensajes[-1].content, sorted(set(_herramientas_usadas(mensajes))), ""
        except Exception as excepcion:
            respuesta, usadas, error = "", [], f"{type(excepcion).__name__}: {str(excepcion)[:150]}"

        if tipo == "memoria":
            dato_ok = any("ana" in recuerdo.lower() for recuerdo in leer_recuerdos(store, usuario))
        else:
            dato_ok = verificar_dato(tipo, respuesta, error)
        filas.append({
            "modelo": modelo, "tipo": tipo, "pregunta": pregunta,
            "tools": usadas, "tools ok": set(usadas) == esperadas and not error, "dato ok": dato_ok,
            "segundos": round(time.time() - inicio, 1), "llamadas LLM": contador.llamadas,
            "tokens entrada": contador.entrada, "tokens salida": contador.salida,
            "error": error, "respuesta": respuesta,
        })
        print(f"[{modelo}] {tipo}: tools={usadas} ({time.time() - inicio:.0f} s) {error}", flush=True)
    return filas


def ejecutar(modelos: list[str] = MODELOS) -> pd.DataFrame:
    """Compara los LLMs en paralelo, con una memoria temporal, y guarda los resultados."""
    config.ARCHIVO_MEMORIA = Path(tempfile.mkdtemp()) / "comparacion.sqlite3"
    from src.agente import construir_agente
    from src.memoria import crear_memoria_corto_plazo, crear_memoria_largo_plazo

    # Las memorias y los agentes se crean antes de lanzar los hilos: inicializar la
    # misma base SQLite desde varios hilos a la vez falla.
    store = crear_memoria_largo_plazo()
    checkpointer = crear_memoria_corto_plazo()
    agentes = [construir_agente(checkpointer, store, modelo=modelo) for modelo in modelos]
    with ThreadPoolExecutor(max_workers=len(modelos)) as hilos:
        resultados = hilos.map(evaluar_modelo, modelos, agentes, [store] * len(modelos))
        filas = [fila for resultado in resultados for fila in resultado]
    ARCHIVO_RESULTADOS.parent.mkdir(exist_ok=True)
    ARCHIVO_RESULTADOS.write_text(
        json.dumps({"fecha": date.today().isoformat(), "resultados": filas}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return pd.DataFrame(filas)


def cargar() -> tuple[str, pd.DataFrame]:
    """Lee los resultados guardados: (fecha de ejecución, tabla)."""
    datos = json.loads(ARCHIVO_RESULTADOS.read_text(encoding="utf-8"))
    tabla = pd.DataFrame(datos["resultados"])
    # Se recalcula con los patrones actuales (la memoria se verificó contra el store).
    es_memoria = tabla["tipo"] == "memoria"
    tabla.loc[~es_memoria, "dato ok"] = [
        verificar_dato(fila.tipo, fila.respuesta, fila.error) for fila in tabla[~es_memoria].itertuples()
    ]
    return datos["fecha"], tabla


def resumen(tabla: pd.DataFrame) -> pd.DataFrame:
    """Una fila por modelo con los totales y promedios de la batería."""
    verificables = tabla[tabla["dato ok"].notna()]
    return pd.DataFrame({
        "tools ok": tabla.groupby("modelo")["tools ok"].sum().astype(int).astype(str) + f" de {len(CASOS)}",
        "dato ok": verificables.groupby("modelo")["dato ok"].sum().astype(int).astype(str)
        + " de " + verificables.groupby("modelo").size().astype(str),
        "errores": tabla.groupby("modelo")["error"].apply(lambda e: (e != "").sum()),
        "segundos (media)": tabla.groupby("modelo")["segundos"].mean().round(1),
        "llamadas LLM": tabla.groupby("modelo")["llamadas LLM"].sum(),
        "tokens entrada": tabla.groupby("modelo")["tokens entrada"].sum(),
        "tokens salida": tabla.groupby("modelo")["tokens salida"].sum(),
    }).reindex(MODELOS)


if __name__ == "__main__":
    tabla = ejecutar()
    pd.set_option("display.width", 200)
    print(resumen(tabla).to_string())
    print(f"Resultados guardados en {ARCHIVO_RESULTADOS}")
