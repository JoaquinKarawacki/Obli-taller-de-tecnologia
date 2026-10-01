# BITÁCORA — Decisiones de diseño y avances

> Registro vivo del proyecto. Cada decisión relevante y cada avance se anota acá,
> con fecha, para poder justificarlo en el informe y en la defensa.
> El plan y las tareas están en [`PLAN.md`](./PLAN.md).

---

## Registro de decisiones (ADR liviano)

Formato: **fecha — decisión — motivo — alternativas descartadas**.

### 2026-09-24 — Orquestación con LangGraph (grafo manual)
- **Decisión:** construir el agente armando el grafo a mano (estado + nodos + edges).
- **Motivo:** la letra prohíbe `create_agent()` / `create_react_agent()` y abstracciones que oculten la lógica.
- **Alternativas descartadas:** helpers prearmados de LangChain/LangGraph.

### 2026-09-24 — Base vectorial: ChromaDB
- **Decisión:** usar Chroma persistente en disco, con colecciones separadas para `dataset` y `papers`.
- **Motivo:** setup simple, local, sin cuenta cloud; suficiente para el volumen del obligatorio.
- **Alternativas descartadas:** Pinecone/Qdrant (más setup), FAISS (sin persistencia cómoda de metadatos).

### 2026-09-24 — LLM: Groq (Llama 3.3 70B)
- **Decisión:** `llama-3.3-70b-versatile` vía Groq (`langchain-groq`).
- **Motivo:** gratis (free tier), muy rápido y con buen tool calling. Mantiene el proyecto sin costo.
- **Alternativas descartadas:** OpenAI (requiere pago), Gemini (opción válida), Ollama (necesita hardware).
- **Requiere:** API key gratuita de Groq en `.env` (`GROQ_API_KEY`).

### 2026-09-24 — Embeddings: sentence-transformers multilingüe (local)
- **Decisión:** `intfloat/multilingual-e5-small` con `sentence-transformers`, corriendo local.
- **Motivo:** Groq no ofrece API de embeddings; este modelo es gratis, local y bueno para **español**.
- **Alternativas / opcional:** comparar con otros modelos multilingües (Fase 8).

### 2026-09-24 — Estructura: módulos .py en vez de notebook
- **Decisión:** el equipo optó por **archivos `.py` separados** (uno por fase) en lugar de un único notebook, por ser más modular y legible y permitir trabajo en paralelo.
- **La letra dice "preferentemente `.ipynb`"** y pide "notebook con outputs" en entregables → **no es obligatorio**. Para cubrirlo, al cierre se agrega un **notebook demo** que importa los módulos y muestra outputs.
- **Alternativa descartada:** todo en un solo notebook (menos legible, difícil de dividir entre integrantes).

### 2026-09-24 — Consolidación del dataset en un único CSV
- **Decisión:** unir los 6 corpus individuales en `data/dataset_consolidado.csv` con columnas `fuente, sender, subject, body, urls, label`.
- **Motivo:** el dataset trae 7 CSV; `phishing_email.csv` ya es una combinación pero **pierde metadatos** (sender, urls). Se arma uno propio conservando lo relevante.
- **Detalles:** se descartan `receiver` y `date` (poco útiles/inconsistentes); Enron y Ling no tienen `sender`/`urls` (quedan NaN); se quitan duplicados por `(subject, body)`.
- **Resultado:** 82.486 filas (42.891 phishing / 39.595 legítimos). Archivo de ~145 MB, regenerable con `python -m scripts.consolidar_dataset` (no se commitea).

### 2026-09-24 — Interfaz: Gradio
- **Decisión:** UI de chat con Gradio para la demo/defensa.
- **Motivo:** opcional que suma; se arma en pocas líneas y es visual.
- **Alternativa descartada:** FastAPI (más trabajo, sin UI directa).

### 2026-09-24 — Fase 2: ingesta del dataset (muestra 10k)
- **Decisión:** indexar una **muestra balanceada de 10.000 correos** (5k phishing / 5k legítimos), **por indicación del profe** (no todo el dataset). Tamaño configurable en `config.py` (`TAMANO_MUESTRA`).
- **Chunking:** `RecursiveCharacterTextSplitter`, `chunk_size=800`, `chunk_overlap=100` → ~33.792 fragmentos.
- **Embeddings:** `intfloat/multilingual-e5-small` (locales, multilingües) con subclase `EmbeddingsE5` que agrega los prefijos `passage:`/`query:` que exige e5.
- **Base vectorial:** ChromaDB persistido en `chroma_db/`, colección `dataset`, metadatos por chunk (fuente, sender, urls, label, id_correo).
- **Indexado por lotes:** se insertan de a 500 fragmentos. **Motivo:** hacerlo todo junto con `Chroma.from_documents` agotó la RAM y el proceso fue matado; por lotes el pico de memoria se mantiene bajo.

### 2026-09-24 — Dataset elegido: Phishing Email Dataset (inglés)
- **Decisión:** usar `naserabdullahalam/phishing-email-dataset` (Kaggle). Temática: **ciberseguridad / correos de phishing**.
- **Motivo:** texto real de correos (asunto + cuerpo + etiqueta phishing/legítimo), ideal para RAG y para un chatbot demostrable ("¿este correo es phishing?", "¿qué patrones usan?").
- **Idioma de los datos:** inglés.
- **Descarga:** vía `kagglehub` (`kagglehub.dataset_download("naserabdullahalam/phishing-email-dataset")`).

### 2026-09-24 — Enfoque multilingüe (datos EN, chatbot ES+EN)
- **Decisión:** los datos están en inglés pero el chatbot interactúa en **español e inglés** (responde en el idioma en que le hablen).
- **Implicancia clave:** se usa **LLM multilingüe** (Llama 3.3 70B, ya elegido) y **embeddings multilingües** (`multilingual-e5-small`, ya elegido) para permitir **retrieval cross-lingual**: una pregunta en español recupera correctamente fragmentos en inglés.
- **Alternativa descartada:** embeddings solo-inglés (romperían las consultas en español).

### 2026-09-24 — Código 100% en español
- **Decisión:** todo el código (variables, funciones, clases, comentarios, docstrings, logs, prompts y textos al usuario) se escribe en **español**.
- **Motivo:** coherencia con el proyecto y claridad para el grupo y la defensa.
- **Excepción:** nombres propios de librerías/APIs se mantienen (`LangGraph`, `thread_id`, etc.).

### 2026-09-24 — Flujo de Git: una rama por tarea → merge a develop
- **Decisión:** trabajar con **una rama por feature/tarea** que sale de `develop` (`feature/...`, `fix/...`, `docs/...`), y al terminar **mergear a `develop`**. `main` es la rama estable de entrega.
- **Motivo:** historial ordenado, trabajo paralelo entre integrantes y `develop` siempre integrable.

### 2026-10-01 — Papers: los 6 PDFs de la materia (colección `papers`)
- **Decisión:** indexar los 6 documentos de "Obligatorio setiembre 2026 - Papers" (Attention, GPT, GPT-3, Scaling Laws, Subliminal Learning y el libro *Designing Data-Intensive Applications*), copiados a `data/papers/`.
- **Cómo:** `pypdf` extrae el texto **página por página** (para guardar el número de página y poder citar), se limpia (guiones de corte de línea, saltos), chunking 800/100 y embeddings e5 → colección `papers`, por lotes. Metadatos: `titulo`, `archivo`, `pagina`.
- **Resultado:** 2.937 fragmentos (2.191 son del libro de Kleppmann). Indexado en ~9 min en CPU.
- **Títulos legibles** en `config.TITULOS_PAPERS`: sirven para citar y para filtrar la búsqueda por paper.

### 2026-10-01 — Fase 3: recuperación como tools + umbral de relevancia
- **Decisión:** adaptar el pipeline del `Demo_vectoriales_crawler` (`search` + `format_message`) a `src/recuperacion.py`: `buscar_fragmentos()` (top-k con puntaje 0-1) y `formatear_fragmentos()` (contexto numerado con la fuente: paper + página, o corpus + etiqueta del correo).
- **Tools para el agente:** `buscar_en_papers` (filtro opcional por paper, con título aproximado), `buscar_en_dataset` (filtro opcional phishing/legítimo), `listar_papers` y `estadisticas_dataset` (conteos con pandas: la búsqueda semántica no sirve para contar).
- **Control de alucinaciones:** si ningún fragmento supera `UMBRAL_RELEVANCIA` la tool devuelve `SIN EVIDENCIA` en vez de fragmentos irrelevantes, y si una colección no está construida devuelve un `ERROR` explícito (no "no hay nada").
- **Umbral = 0.68** (calibrado, ver "Desafíos"). Es un piso; el router y el prompt del generador completan el filtrado.

### 2026-10-01 — Reutilizar el Lab03 (RAG Pipeline) en el pipeline de papers
- **Decisión:** alinear la ingesta y la recuperación con lo visto en el Lab03, para que el código sea el del curso y fácil de defender:
  - `PyPDFLoader` (un `Document` por página) + `normalize_text` (→ `normalizar_texto`, preprocesamiento mínimo) + `split_documents`. Se verificó que el resultado es **idéntico** al índice que ya estaba construido (2.937 fragmentos), así que no hubo que reindexar.
  - `LongContextReorder` sobre los fragmentos recuperados (problema "Lost in the Middle"): los más relevantes van a los extremos del contexto.
  - Filtros por metadata (Parte 3 del lab), pero con metadata **real** (`titulo`, `pagina`, `label`) en vez de sintética.
  - Comparación de estrategias de chunking (Parte 2), ampliada a un experimento que mide calidad de recuperación (ver "Experimentos opcionales").
  - Pregunta con contexto vs. sin contexto (Parte 1): queda para el notebook demo (necesita el LLM).
- **Nota:** `PyPDFLoader` y `LongContextReorder` vienen de `langchain-community`, que LangChain marcó como *sunset* (sin mantenimiento activo). Funciona; si en el futuro se rompe, ambos son triviales de reemplazar (`pypdf` directo y un reordenamiento de 5 líneas).

### 2026-10-01 — LLM: cambio a Qwen 3.8 27B (Groq)
- **Decisión:** usar `qwen/qwen3.8-27b` vía Groq. Se mantiene Groq como único proveedor (sin alternativa, por simplicidad).
- **Motivo:** `llama-3.3-70b-versatile` **ya no existe en Groq** (error 404 `model_not_found`). Modelos de chat disponibles: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`.
- **Prueba de tool calling** con las tools del proyecto y preguntas en español: ambos candidatos eligieron bien la tool (papers / estadísticas) y respondieron el small-talk sin buscar; `gpt-oss-120b` le pasó un argumento inventado a una tool sin parámetros, Qwen no cometió errores → se elige Qwen.

### 2026-10-01 — Chroma (local) en vez de Pinecone
- **Decisión:** mantener **ChromaDB local** aunque el Lab03 use Pinecone.
- **Motivo:** no requiere cuenta ni API key (los docentes pueden correrlo sin nuestras credenciales), funciona sin internet (la defensa no depende de la red ni de la cuota del plan gratuito de Pinecone) y el índice persiste en una carpeta (`chroma_db/`, ~26 MB con los papers) que puede reutilizarse sin reindexar. La letra admite cualquiera ("Pinecone, Qdrant, Chroma, FAISS").
- **Alternativa descartada:** Pinecone — su ventaja (índice en la nube, sin reindexar) se cubre guardando `chroma_db/`.

---

## Temática / dataset — DECIDIDO

- ✅ **Dataset:** `naserabdullahalam/phishing-email-dataset` — Phishing Email Dataset (Kaggle).
- **Temática:** ciberseguridad / correos de phishing. **Idioma:** inglés.
- **Chatbot:** multilingüe (ES + EN).

### Opciones evaluadas antes (descartadas)
| Género | Dataset (Kaggle) | Motivo de descarte |
|--------|------------------|--------------------|
| 📰 Noticias ES | Spanish News Classification | Se optó por temática de ciberseguridad |
| 🎬 Cine ES | IMDB 50K Movie Reviews (Spanish) | ídem |
| 🏨 Hoteles ES | Andalusian Hotels' Reviews | ídem |
| 🩺 Enfermedades ES | — | Texto médico en español escaso en Kaggle |

---

## Diario de avances

### 2026-09-24
- Repo clonado y letra del obligatorio subida a `main`.
- Creada rama `develop` para el trabajo.
- Definido stack tecnológico inicial (ver `PLAN.md` §3).
- Creados `PLAN.md` (plan + fases + stack) y `BITACORA.md` (este archivo).
- Definidas convenciones: código 100% en español y flujo de ramas (feature → develop).
- Fase 0 (setup): README, esqueleto del notebook y estructura de carpetas.
- **Dataset cerrado:** Phishing Email Dataset (inglés) + enfoque multilingüe (chatbot ES+EN).
- Agregado `kagglehub` a dependencias.
- **Cambio de estructura:** de notebook a módulos `.py` (uno por fase) + `main.py`.
- **Dataset consolidado:** `src/datos.py` + `scripts/consolidar_dataset.py` generan `data/dataset_consolidado.csv` (82.486 filas). Módulos esqueleto creados para Fases 2–6.
- Entorno virtual `venv` creado; instalados kagglehub, pandas y python-dotenv.
- **Fase 2 (ingesta):** `src/ingesta.py` + `scripts/construir_indice.py` — muestra 10k balanceada, chunking, embeddings e5 y Chroma (`chroma_db/`, colección `dataset`) indexado por lotes. Instaladas sentence-transformers, chromadb, langchain-chroma, langchain-huggingface, langchain-text-splitters.

### 2026-10-01
- **Fase 2 (papers):** `construir_indice_papers()` + `scripts/construir_indice_papers.py`. 2.937 fragmentos en la colección `papers`. Retrieval cross-lingual verificado (preguntas en español → página correcta del paper en inglés).
- **Fase 3 (recuperación):** `src/recuperacion.py` con búsqueda, formateo con fuentes y 4 tools. `scripts/probar_recuperacion.py` para calibrar el umbral.
- Probado con Python 3.14 (todas las dependencias instalan bien).

---

## Desafíos y soluciones
*(se completa a medida que aparezcan)*

| Desafío | Solución aplicada |
|---------|-------------------|
| La búsqueda siempre devuelve "algo", aunque la pregunta no tenga nada que ver con los documentos (ej. "receta de lasaña") → riesgo de alucinar con contexto irrelevante. | Umbral de relevancia: debajo de él la tool devuelve `SIN EVIDENCIA` y el agente debe admitir que no sabe. |
| Los puntajes de e5 están "comprimidos" y se **solapan**: consultas pertinentes 0.72–0.83, ajenas 0.63–0.75 ("hola, ¿cómo estás?" dio 0.75). Un umbral solo no separa bien. | Umbral como **piso** (0.68: corta lo claramente ajeno sin perder preguntas válidas) + defensa en capas: el router no busca en small-talk y el generador verifica que los fragmentos respondan. |
| Todas las consultas ajenas caían en la misma **tabla de números** de GPT-3 (pág. 63): los fragmentos sin texto real quedan "cerca de todo" en el espacio vectorial. | Filtro en la ingesta: se descartan fragmentos con < 50% de letras (34 de 2.971, casi todos tablas). Las tablas con contenido útil (Tabla 2 de Attention, parámetros de GPT-3) quedan por encima y se conservan. |
| Las **bibliografías** de los papers también atraen consultas ajenas ("mundial de fútbol 2022" → referencias de GPT-3 con relevancia 0.69, apenas sobre el umbral). | **No se filtran** (decisión consciente): una heurística por años/"et al."/"arXiv" también marcaba párrafos con contenido real. Se cubre con las otras capas (router + prompt del generador que verifica que el fragmento responda). Queda como mejora posible. |
| Sin el índice del dataset construido, Chroma crea una colección vacía y la búsqueda decía "sin evidencia", lo que el agente interpretaría como "no hay correos así". | La búsqueda detecta la colección vacía y la tool devuelve un `ERROR` explícito. |

---

## Experimentos opcionales

### 2026-10-01 — Chunking × embeddings (`scripts/experimento_chunking_embeddings.py`)
- **Qué se midió:** 9 preguntas sobre los 4 papers cortos, cada una en español y en inglés. Para cada pregunta se conoce una frase del paper que contiene la respuesta; hay acierto si está en alguno de los 4 fragmentos recuperados (acierto@4). MRR = promedio de 1/posición del primer fragmento correcto. Chroma en memoria.
- **Estrategias:** las 3 del Lab03 (250/30, 600/100, 1200/200) + la del proyecto (800/100). **Modelos:** `all-MiniLM-L6-v2` (Lab02/03, entrenado en inglés) vs `multilingual-e5-small` (proyecto).

| Embeddings | Chunking | Fragmentos | Indexado (s) | Acierto@4 ES | MRR ES | Acierto@4 EN | MRR EN |
|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 250/30 | 1202 | 14 | 22% | 0.15 | 56% | 0.43 |
| all-MiniLM-L6-v2 | 600/100 | 549 | 14 | 11% | 0.06 | 56% | 0.47 |
| all-MiniLM-L6-v2 | 1200/200 | 287 | 9 | 11% | 0.06 | 67% | 0.30 |
| all-MiniLM-L6-v2 | 800/100 | 400 | 12 | 11% | 0.03 | 44% | 0.23 |
| multilingual-e5-small | 250/30 | 1202 | 26 | 33% | 0.26 | 33% | 0.33 |
| multilingual-e5-small | 600/100 | 549 | 28 | 44% | 0.25 | 67% | 0.46 |
| multilingual-e5-small | 1200/200 | 287 | 28 | 56% | 0.34 | 56% | 0.42 |
| multilingual-e5-small | 800/100 | 400 | 26 | 33% | 0.20 | 44% | 0.44 |

- **Conclusiones:**
  - **e5 multilingüe es claramente mejor en español** (33–56% vs 11–22%): MiniLM casi no recupera nada si la pregunta está en español y el paper en inglés. Confirma con datos la decisión de embeddings (chatbot ES+EN sobre datos en inglés).
  - En inglés ambos modelos rinden parecido (44–67%): la ventaja de e5 es el cross-lingual.
  - MiniLM indexa ~2x más rápido (modelo más chico, 384 dims ambos).
  - **Chunking:** los chunks muy chicos (250) empeoran con e5; 600 y 1200 rindieron mejor que 800. Pero con 9 preguntas cada una vale 11%, así que las diferencias entre tamaños **no son concluyentes**. No se cambió 800/100 (cambiarlo implica reindexar dataset y papers); queda como mejora: ampliar la batería de preguntas y re-evaluar 600 vs 800 vs 1200.
  - La métrica es estricta (exige la frase exacta en el fragmento), por eso los porcentajes absolutos son bajos; sirve para **comparar** configuraciones, no como precisión real del chatbot.

---

## Prompts y prompt engineering
*(guardar acá los prompts clave: router, generación, control de alucinaciones, extracción de memoria)*

---

## Uso de IA generativa (para citar en el informe)
- **Herramienta:** Claude Code (Anthropic).
- **Contexto de uso:** apoyo en la estructuración del proyecto, redacción de documentación (`PLAN.md`, `BITACORA.md`) y exploración de datasets.
- **Responsabilidad:** todo el contenido generado con IA se revisa y verifica; los errores son responsabilidad del grupo.
