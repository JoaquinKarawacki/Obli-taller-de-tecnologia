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
import re
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
from pypdf import PdfReader
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


@lru_cache(maxsize=1)
def crear_embeddings() -> EmbeddingsE5:
    """Prende el traductor de embeddings (una sola vez por proceso).

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


# ---------------------------------------------------------------------------
# Papers (documentos largos en PDF)
# ---------------------------------------------------------------------------
# Mismo proceso que con los correos, pero la fuente son PDFs:
#
#     PDF -> texto de cada página -> fragmentos (chunking) -> embeddings -> Chroma
#
# Se trabaja página por página para poder guardar el número de página como
# metadato: así el chatbot puede citar "según <paper>, pág. N".


def _limpiar_texto_pdf(texto: str) -> str:
    """Arregla los defectos típicos del texto extraído de un PDF.

    - Une las palabras cortadas con guion al final de línea ("trans-\\nformer").
    - Convierte los saltos de línea simples en espacios (en el PDF cortan la
      oración a la mitad) y colapsa los espacios repetidos.
    """
    texto = re.sub(r"-\n(\w)", r"\1", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def es_fragmento_util(texto: str) -> bool:
    """Descarta fragmentos que son casi todo números/símbolos (tablas de resultados).

    Problema detectado al calibrar: esos fragmentos no tienen significado propio y
    sus embeddings quedan "cerca de todo", así que aparecían como mejor resultado
    para preguntas que no tienen nada que ver (ej. "receta de lasaña" -> tabla de
    BLEU de GPT-3). Se exige que al menos la mitad de los caracteres sean letras.
    """
    letras = sum(caracter.isalpha() for caracter in texto)
    return letras / max(len(texto), 1) >= config.PROPORCION_MIN_LETRAS


def _paper_a_documentos(ruta_pdf: Path, separador) -> list[Document]:
    """Convierte un PDF en fragmentos (Document) con título y página."""
    titulo = config.TITULOS_PAPERS.get(ruta_pdf.name, ruta_pdf.stem)
    lector = PdfReader(ruta_pdf)

    documentos: list[Document] = []
    for numero, pagina in enumerate(lector.pages, start=1):
        texto = _limpiar_texto_pdf(pagina.extract_text() or "")
        if len(texto) < 50:
            continue  # páginas en blanco, solo imágenes o casi vacías

        metadatos = {"titulo": titulo, "archivo": ruta_pdf.name, "pagina": numero}
        for fragmento in separador.split_text(texto):
            if es_fragmento_util(fragmento):
                documentos.append(Document(page_content=fragmento, metadata=metadatos))
    return documentos


def construir_indice_papers(rutas_pdf: list[Path] | None = None, tam_lote: int = 500) -> Chroma:
    """Arma y guarda el índice vectorial de los papers (colección ``papers``).

    Args:
        rutas_pdf: PDFs a indexar. Por defecto, todos los de ``data/papers/``.
        tam_lote: fragmentos que se insertan por vez (igual que con el dataset,
            para no agotar la RAM).
    """
    inicio = time.time()
    if rutas_pdf is None:
        rutas_pdf = sorted(config.DIR_PAPERS.glob("*.pdf"))
    if not rutas_pdf:
        raise FileNotFoundError(f"No hay PDFs en {config.DIR_PAPERS}")

    separador = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    print("Leyendo y fragmentando papers...")
    documentos: list[Document] = []
    for ruta in rutas_pdf:
        docs_paper = _paper_a_documentos(ruta, separador)
        print(f"  {docs_paper[0].metadata['titulo'] if docs_paper else ruta.name}: {len(docs_paper)} fragmentos")
        documentos.extend(docs_paper)
    total = len(documentos)
    print(f"  total de fragmentos: {total}")

    print("Creando embeddings e indexando en Chroma por lotes...")
    config.DIR_CHROMA.mkdir(exist_ok=True)
    indice = cargar_indice(config.COLECCION_PAPERS)
    try:
        indice.reset_collection()  # se limpia para no duplicar al re-indexar
    except Exception:
        pass

    for inicio_lote in range(0, total, tam_lote):
        indice.add_documents(documentos[inicio_lote:inicio_lote + tam_lote])
        print(f"  indexados {min(inicio_lote + tam_lote, total)}/{total}")

    print(f"Listo en {time.time() - inicio:.1f} s. Índice guardado en {config.DIR_CHROMA}")
    return indice


def cargar_indice(coleccion: str) -> Chroma:
    """Reabre una colección de Chroma ya guardada (``dataset`` o ``papers``)."""
    return Chroma(
        collection_name=coleccion,
        embedding_function=crear_embeddings(),
        persist_directory=str(config.DIR_CHROMA),
    )
