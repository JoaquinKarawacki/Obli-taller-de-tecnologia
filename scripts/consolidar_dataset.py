"""Ejecutable: genera ``data/dataset_consolidado.csv`` a partir de los corpus.

Uso (desde la raíz del repo):
    python -m scripts.consolidar_dataset
"""
from src.datos import consolidar_dataset

if __name__ == "__main__":
    df = consolidar_dataset(guardar=True)

    print("\nVista previa:")
    print(df.head())

    print("\nDistribución de etiquetas (label):")
    print(df["label"].value_counts(dropna=False))

    print("\nFilas por fuente:")
    print(df["fuente"].value_counts())
