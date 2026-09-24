"""Fase 2 — Ingesta y embeddings.

Este módulo convierte los correos del dataset en algo que se puede "buscar por
significado". El proceso completo es:

    correos (CSV)
      -> tomar una muestra balanceada de 10.000 (5k phishing + 5k legítimos)
      -> por cada correo, juntar asunto + cuerpo en un solo texto
      -> cortar ese texto en fragmentos (chunking)
      -> convertir cada fragmento en un vector (embedding)
      -> guardar los vectores en ChromaDB (la base de datos vectorial)

¿Por qué vectores? La computadora no entiende texto, entiende números. Un
"embedding" es la huella numérica del significado de un texto: dos textos que
significan cosas parecidas quedan con vectores cercanos, aunque usen palabras
distintas e incluso en otro idioma (por eso se puede preguntar en español sobre
correos en inglés).

Los embeddings son locales (modelo e5 multilingüe), así que esta fase no necesita
la API key de Groq.
"""
import time

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src import datos


class EmbeddingsE5(HuggingFaceEmbeddings):
    """El "traductor" de texto a vectores.

    Es solo el modelo que convierte texto -> números; no sabe de dónde sale el
    texto. El modelo e5 exige un prefijo distinto según qué se traduzca:
      - "passage: " para los documentos (los correos que guardamos)
      - "query: "   para las consultas (lo que pregunta el usuario)
    Esta subclase agrega esos prefijos automáticamente para no olvidarnos nunca.
    """

    def embed_documents(self, textos: list[str]) -> list[list[float]]:
        # Se llama al indexar los correos: antepone "passage: " a cada fragmento.
        textos_con_prefijo = [config.PREFIJO_PASSAGE + t for t in textos]
        return super().embed_documents(textos_con_prefijo)

    def embed_query(self, texto: str) -> list[float]:
        # Se llama al buscar: antepone "query: " a la pregunta del usuario.
        return super().embed_query(config.PREFIJO_QUERY + texto)


def crear_embeddings() -> EmbeddingsE5:
    """Prende el traductor de embeddings.

    La primera vez descarga el modelo desde internet (~470 MB); después queda
    cacheado en la máquina y ya no se vuelve a bajar.
    """
    return EmbeddingsE5(model_name=config.MODELO_EMBEDDINGS)


def _muestra_balanceada(df: pd.DataFrame) -> pd.DataFrame:
    """Elige 10.000 correos parejos: mitad phishing (label=1), mitad legítimos (0).

    - Primero descarta los correos sin cuerpo (no aportan nada para indexar).
    - Toma 5.000 de cada clase (TAMANO_MUESTRA // 2).
    - Usa una semilla fija para que la muestra sea siempre la misma (reproducible).
    - Mezcla el resultado para que no queden todos los phishing juntos.
    """
    # Nos quedamos solo con las filas que tienen cuerpo (body no vacío).
    df = df[df["body"].notna() & (df["body"].astype(str).str.strip() != "")]

    por_clase = config.TAMANO_MUESTRA // 2  # 5.000 por clase
    partes = []
    for etiqueta in (0, 1):  # 0 = legítimo, 1 = phishing
        subconjunto = df[df["label"] == etiqueta]
        # min() por si alguna clase tuviera menos de 5.000 correos disponibles.
        n = min(por_clase, len(subconjunto))
        partes.append(subconjunto.sample(n=n, random_state=config.SEMILLA))

    # Unimos las dos clases y mezclamos todo (frac=1 = barajar el 100%).
    muestra = pd.concat(partes).sample(frac=1, random_state=config.SEMILLA).reset_index(drop=True)
    return muestra


def _texto_del_correo(fila: pd.Series) -> str:
    """Junta asunto + cuerpo de un correo en un único texto.

    Queda algo como:
        Asunto: <asunto>

        <cuerpo>
    Maneja los casos en que asunto o cuerpo vengan vacíos (NaN).
    """
    asunto = "" if pd.isna(fila["subject"]) else str(fila["subject"]).strip()
    cuerpo = "" if pd.isna(fila["body"]) else str(fila["body"]).strip()
    if asunto:
        return f"Asunto: {asunto}\n\n{cuerpo}"
    return cuerpo


def _a_documentos(df: pd.DataFrame) -> list[Document]:
    """Convierte los correos en fragmentos (Document) listos para indexar.

    Por cada correo: arma el texto, lo CORTA en pedazos de ~800 caracteres
    (chunking) y a cada pedazo le adjunta metadatos (de qué corpus vino, el
    remitente, si tenía urls y si es phishing o no). Esos metadatos después sirven
    para filtrar o mostrar información en las respuestas.
    """
    # El splitter corta textos largos respetando párrafos/oraciones cuando puede.
    separador = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,       # tamaño máximo de cada fragmento
        chunk_overlap=config.CHUNK_OVERLAP,  # se repiten algunos caracteres entre fragmentos
    )                                        # para no cortar ideas justo al medio

    documentos: list[Document] = []
    for id_correo, fila in df.iterrows():
        texto = _texto_del_correo(fila)
        if not texto:
            continue  # correo vacío, lo salteamos

        # Metadatos que viajan pegados a cada fragmento de este correo.
        metadatos = {
            "id_correo": int(id_correo),
            "fuente": str(fila["fuente"]),
            "sender": "" if pd.isna(fila["sender"]) else str(fila["sender"]),
            "urls": "" if pd.isna(fila["urls"]) else str(fila["urls"]),
            "label": int(fila["label"]),  # 0 = legítimo, 1 = phishing
        }

        # Un correo puede generar varios fragmentos; cada uno es un Document.
        for fragmento in separador.split_text(texto):
            documentos.append(Document(page_content=fragmento, metadata=metadatos))

    return documentos


def construir_indice_dataset(tam_lote: int = 500) -> Chroma:
    """Orquestador de la Fase 2: arma y guarda el índice vectorial del dataset.

    Encadena todos los pasos: cargar el CSV -> muestra balanceada -> chunking ->
    embeddings -> guardar en la colección ``dataset`` dentro de ``chroma_db/``.
    Es la función que se ejecuta desde ``scripts/construir_indice.py``.

    La vectorización se hace POR LOTES (``tam_lote`` fragmentos por vez) en vez de
    todo junto: así el pico de memoria RAM se mantiene bajo (procesar 33k
    fragmentos de una sola vez agota la memoria) y además se ve el progreso.
    """
    inicio = time.time()

    print("Cargando dataset consolidado...")
    df = datos.cargar_consolidado()

    print(f"Tomando muestra balanceada de {config.TAMANO_MUESTRA} correos...")
    muestra = _muestra_balanceada(df)
    print(f"  correos en la muestra: {len(muestra)}")
    print(f"  distribución (label): {muestra['label'].value_counts().to_dict()}")

    print("Generando fragmentos (chunking)...")
    documentos = _a_documentos(muestra)
    total = len(documentos)
    print(f"  fragmentos generados: {total}")

    print("Creando embeddings e indexando en Chroma por lotes...")
    config.DIR_CHROMA.mkdir(exist_ok=True)

    # Colección persistida (vacía). Se llena de a lotes más abajo.
    indice = Chroma(
        collection_name=config.COLECCION_DATASET,
        embedding_function=crear_embeddings(),
        persist_directory=str(config.DIR_CHROMA),
    )
    # Si quedó algo de una corrida anterior, se limpia para no duplicar.
    try:
        indice.reset_collection()
    except Exception:
        pass

    # Se insertan los fragmentos en tandas de `tam_lote` (bajo consumo de memoria).
    for inicio_lote in range(0, total, tam_lote):
        lote = documentos[inicio_lote:inicio_lote + tam_lote]
        indice.add_documents(lote)
        print(f"  indexados {min(inicio_lote + tam_lote, total)}/{total}")

    print(f"Listo en {time.time() - inicio:.1f} s. Índice guardado en {config.DIR_CHROMA}")
    return indice


def cargar_indice_dataset() -> Chroma:
    """Reabre la colección ``dataset`` ya guardada, sin recalcular nada.

    Lo usará la Fase 3 (búsqueda) para no tener que reconstruir el índice cada vez.
    """
    return Chroma(
        collection_name=config.COLECCION_DATASET,
        embedding_function=crear_embeddings(),
        persist_directory=str(config.DIR_CHROMA),
    )


def construir_indice_papers(rutas_pdf):
    """Procesa papers (PDF), los chunkea y los guarda en Chroma."""
    # TODO (fase posterior): leer PDFs + chunking + embeddings + colección papers
    raise NotImplementedError
