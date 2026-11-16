"""Fase 1 — Carga y consolidación del dataset de correos de phishing.

Une los corpus individuales descargados de Kaggle (carpeta ``data/``) en un
único CSV con las columnas más relevantes para el sistema RAG.
"""
import pandas as pd

from src import config

# Columnas que se conservan de cada corpus (las que falten quedan como NaN)
_COLUMNAS_BASE = ["sender", "subject", "body", "urls", "label"]


def consolidar_dataset(guardar: bool = True) -> pd.DataFrame:
    """Une los corpus de ``data/`` en un único DataFrame consolidado.

    - Conserva solo las columnas relevantes: ``sender, subject, body, urls, label``.
    - Agrega la columna ``fuente`` con el nombre del corpus de origen.
    - Rellena con NaN las columnas ausentes (Enron y Ling no traen sender/urls).
    - Elimina duplicados exactos por ``(subject, body)``.

    Args:
        guardar: si es True, escribe el resultado en ``data/dataset_consolidado.csv``.

    Returns:
        El DataFrame consolidado.
    """
    marcos = []
    for archivo in config.ARCHIVOS_CORPUS:
        ruta = config.DIR_DATOS / archivo
        df = pd.read_csv(ruta)
        fuente = archivo.replace(".csv", "")

        # Asegura que existan todas las columnas base (las ausentes quedan como NaN)
        for columna in _COLUMNAS_BASE:
            if columna not in df.columns:
                df[columna] = pd.NA

        df = df[_COLUMNAS_BASE].copy()
        df.insert(0, "fuente", fuente)
        marcos.append(df)
        print(f"  {fuente}: {len(df)} filas")

    consolidado = pd.concat(marcos, ignore_index=True)

    filas_antes = len(consolidado)
    consolidado = consolidado.drop_duplicates(subset=["subject", "body"]).reset_index(drop=True)
    print(f"Total: {filas_antes} filas -> {len(consolidado)} tras quitar duplicados")

    if guardar:
        config.DIR_DATOS.mkdir(exist_ok=True)
        consolidado.to_csv(config.CSV_CONSOLIDADO, index=False)
        print(f"Guardado en: {config.CSV_CONSOLIDADO}")

    return consolidado


def cargar_consolidado() -> pd.DataFrame:
    """Carga el dataset ya consolidado desde disco.

    Si no existe, lo genera llamando a :func:`consolidar_dataset`.
    """
    if config.CSV_CONSOLIDADO.exists():
        return pd.read_csv(config.CSV_CONSOLIDADO)
    return consolidar_dataset(guardar=True)
