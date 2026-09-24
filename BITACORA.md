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

---

## Desafíos y soluciones
*(se completa a medida que aparezcan)*

| Desafío | Solución aplicada |
|---------|-------------------|
| — | — |

---

## Prompts y prompt engineering
*(guardar acá los prompts clave: router, generación, control de alucinaciones, extracción de memoria)*

---

## Uso de IA generativa (para citar en el informe)
- **Herramienta:** Claude Code (Anthropic).
- **Contexto de uso:** apoyo en la estructuración del proyecto, redacción de documentación (`PLAN.md`, `BITACORA.md`) y exploración de datasets.
- **Responsabilidad:** todo el contenido generado con IA se revisa y verifica; los errores son responsabilidad del grupo.
