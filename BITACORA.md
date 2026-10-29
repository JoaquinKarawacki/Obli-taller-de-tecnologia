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

### 2026-10-29 — Fase 4: grafo del agente (loop agente ↔ herramientas, escrito a mano)
- **Decisión:** grafo de 2 nodos, `agente` (LLM con las tools bindeadas) y `herramientas` (ejecuta las tools pedidas), con un edge condicional `decidir_siguiente`: si el LLM pidió tools va a `herramientas` y vuelve al `agente`; si no, termina. Es el patrón ReAct, pero armado a mano como en la Parte 3/4 del Lab04.
- **Motivo:** la letra prohíbe `create_react_agent()` y abstracciones que oculten la lógica. Lo que sí pide es que el agente decida dinámicamente el tipo de consulta y use tool calling: este loop lo cumple con la menor cantidad de piezas, y cada una se puede explicar en la defensa.
- **Sin `ToolNode` ni `tools_condition`** (los usa el Lab04): vienen de `langgraph.prebuilt`, el mismo módulo que `create_react_agent`. Escribirlos a mano son ~10 líneas y evita la discusión de si cuentan como "abstracción de alto nivel".
- **Sin nodo router aparte:** el propio LLM decide (guiado por el prompt de sistema) si responde directo (small-talk, preguntas generales) o llama a tools (papers, dataset). Donde antes en esta bitácora se habla del "router", ese rol lo cumple el nodo `agente`. Un router con una llamada extra al LLM duplicaría el consumo de tokens (ver límite de Groq abajo) sin aportar a lo que pide la letra.
- **Prompt de sistema** fuera del estado: se antepone en cada llamada al LLM, así no se repite en el historial que guardará la memoria (Fase 5).
- **`construir_agente(checkpointer, store)`** ya recibe la memoria como parámetro, para enchufar la Fase 5 sin tocar el grafo.

### 2026-10-29 — LLM: sin razonamiento + reintentos (límite de Groq)
- **Problema:** el plan gratuito de Groq limita `qwen/qwen3.8-27b` a **1000 tokens de salida por minuto**; la batería de pruebas cortó con error 429. Qwen 3 además "piensa" antes de responder y esos tokens cuentan para el límite.
- **Decisión:** `reasoning_effort="none"` (sin razonamiento), `max_retries=6` (el cliente de Groq espera lo que indica la API y reintenta) y pedir respuestas concisas en el prompt. Con eso la batería completa corre sin errores y la elección de tools sigue siendo correcta.
- **Alternativa descartada:** otro proveedor o un LLM de respaldo (decisión del equipo: solo Groq, simple).

### 2026-10-29 — Fase 5: memoria en SQLite (checkpointer + store) y tool `guardar_recuerdo`
- **Corto plazo:** `SqliteSaver` (checkpointer de LangGraph). Guarda el estado de cada conversación por `thread_id` después de cada paso. Mismo `thread_id` → se retoma donde quedó; otro → conversación nueva e independiente.
- **Largo plazo:** `SqliteStore` (store de LangGraph), namespace `("usuarios", user_id, "recuerdos")`. Compartido entre todas las conversaciones del usuario.
- **Qué guardar lo decide el agente:** tool `guardar_recuerdo(recuerdo)`. El prompt le indica guardar datos duraderos del usuario (nombre, trabajo, intereses, preferencias, nivel), uno por llamada, sin repetir y nunca datos sensibles.
- **Cuándo usarlo:** en cada turno el nodo `agente` lee los recuerdos del usuario y los agrega al prompt de sistema. Son pocos por usuario, así que se pasan todos (no hace falta búsqueda semántica sobre la memoria).
- **Seguridad entre usuarios:** `user_id` y `store` son argumentos `InjectedToolArg`: el LLM no los ve ni los completa, los agrega el nodo `herramientas` desde el config de la sesión. Así un usuario no puede escribir en la memoria de otro.
- **Sesiones en paralelo:** cada invocación lleva `{"configurable": {"thread_id", "user_id"}}`; probado con dos usuarios en hilos simultáneos.
- **Por qué SQLite y no `MemorySaver`/`InMemoryStore`:** la letra pide "pausar y retomar sin pérdida de contexto"; en RAM se pierde todo al cerrar el programa. SQLite es un archivo local (`memoria.sqlite3`, no se commitea), sin servidor, y viene como paquete oficial de LangGraph (`langgraph-checkpoint-sqlite`).
- **`construir_agente(checkpointer, store)`** ahora los pide obligatorios: un solo camino de código. Las pruebas que no deben ensuciar la memoria (`probar_agente.py`) pasan las versiones en RAM.
- **Mejora posible (no implementada):** recortar el historial de conversaciones muy largas (hoy se manda completo al LLM).

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

### 2026-10-29
- **Fase 4 (agente):** `src/agente.py` con el grafo manual (`agente` ↔ `herramientas`) y el prompt de sistema. `scripts/probar_agente.py` corre una batería de 7 consultas y muestra qué tools eligió el agente:
  - small-talk ("Hola! cómo andás?") → sin tools ✔
  - papers en ES y EN → `buscar_en_papers` filtrando por paper, responde citando páginas y en el idioma de la pregunta ✔
  - "¿Cuántos correos de phishing hay?" → `estadisticas_dataset` (42.891 de 82.486) ✔
  - ejemplo de phishing bancario → `buscar_en_dataset` dos veces (reformula en inglés); admite que no hay uno que pida *explícitamente* la contraseña del banco y muestra los más cercanos ✔
  - "capital de Francia" → sin tools, conocimiento general ✔ (no aclaró que no sale de los documentos; ajuste menor de prompt pendiente)
  - "¿Qué dice Attention sobre el mundial 2022?" → busca y admite que no tiene información ✔
- **Índice del dataset:** se lanzó la construcción local (`python -m scripts.construir_indice`): 33.792 fragmentos de la muestra de 10k correos. Con embeddings en CPU tarda bastante (~1 h); la búsqueda funciona aunque esté a medias, con los fragmentos ya indexados.

- **Fase 5 (memoria):** `src/memoria.py` + agente actualizado. `scripts/probar_memoria.py` verifica:
  - Ana se presenta → el agente guarda "Se llama Ana y trabaja en el área de seguridad informática de un banco" ✔
  - conversación nueva de Ana → recuerda nombre y trabajo (largo plazo) pero no lo que preguntó en la otra (conversaciones independientes) ✔
  - se recrea el agente (simula reiniciar el programa) y Ana retoma la conversación 1 → recuerda que preguntó por multi-head attention (corto plazo persistido) ✔
  - Ana y Beto en paralelo → Beto no sabe nada de Ana y viceversa ✔
- Ajuste de prompt: en preguntas generales ahora aclara "esto es conocimiento general, no sale de los documentos cargados" ✔

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
| El plan gratuito de Groq corta con error 429 (1000 tokens de salida/min para Qwen) y Qwen 3 gasta tokens "pensando" antes de responder. | `reasoning_effort="none"`, `max_retries=6` (espera y reintenta) y respuestas concisas por prompt. |
| SQLite: `database is locked` / `cannot start a transaction within a transaction` al usar checkpointer y store sobre el mismo archivo. | Conexión en modo autocommit (`isolation_level=None`, como hace `SqliteStore.from_conn_string`): cada uno maneja sus propias transacciones. |
| Buscar en la colección `dataset` mientras el índice se está construyendo en otro proceso da `chromadb InternalError: Error finding id`. | Es concurrencia de escritura/lectura de Chroma, no del agente: el índice se construye una sola vez antes de usar el chatbot. |

---

## Prompts y prompt engineering
*(guardar acá los prompts clave: router, generación, control de alucinaciones, extracción de memoria)*

### Prompt de sistema del agente (`src/agente.py`, `PROMPT_SISTEMA`)
Técnicas usadas:
- **Rol** explícito (experto en phishing y en los papers).
- **Descripción de cada tool y cuándo usarla** (complementa los docstrings de las tools, que son lo que el LLM lee para elegir).
- **Clasificación de la consulta por instrucciones**: small-talk → sin tools; papers/dataset → tool obligatoria antes de responder; general → conocimiento propio aclarando que no viene de los documentos.
- **Control de alucinaciones**: responder solo con los fragmentos; ante `SIN EVIDENCIA` o fragmentos que no responden, decir que no hay información; ante `ERROR`, avisar; citar fuente (paper + página o corpus).
- **Idioma espejo**: responder en el idioma del usuario aunque los documentos estén en inglés.
- **Concisión** (también por el límite de tokens de Groq).
- **Memoria guiada por reglas**: qué guardar (datos duraderos del usuario), qué no (preguntas puntuales, datos sensibles) y para qué usarlo (personalizar). Los recuerdos se inyectan al final del prompt en cada turno.

---

## Uso de IA generativa (para citar en el informe)
- **Herramienta:** Claude Code (Anthropic).
- **Contexto de uso:** apoyo en la estructuración del proyecto, redacción de documentación (`PLAN.md`, `BITACORA.md`) y exploración de datasets.
- **Responsabilidad:** todo el contenido generado con IA se revisa y verifica; los errores son responsabilidad del grupo.
