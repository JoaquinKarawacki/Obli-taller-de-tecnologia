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

### 2026-09-24 — Interfaz: Gradio
- **Decisión:** UI de chat con Gradio para la demo/defensa.
- **Motivo:** opcional que suma; se arma en pocas líneas y es visual.
- **Alternativa descartada:** FastAPI (más trabajo, sin UI directa).

### 2026-09-24 — Idioma del dataset: español (a confirmar)
- **Decisión provisoria:** buscar dataset con **texto en español**.
- **Estado:** en evaluación de temática (ver más abajo).

### 2026-09-24 — Código 100% en español
- **Decisión:** todo el código (variables, funciones, clases, comentarios, docstrings, logs, prompts y textos al usuario) se escribe en **español**.
- **Motivo:** coherencia con el proyecto y claridad para el grupo y la defensa.
- **Excepción:** nombres propios de librerías/APIs se mantienen (`LangGraph`, `thread_id`, etc.).

### 2026-09-24 — Flujo de Git: una rama por tarea → merge a develop
- **Decisión:** trabajar con **una rama por feature/tarea** que sale de `develop` (`feature/...`, `fix/...`, `docs/...`), y al terminar **mergear a `develop`**. `main` es la rama estable de entrega.
- **Motivo:** historial ordenado, trabajo paralelo entre integrantes y `develop` siempre integrable.

---

## Temática / dataset — opciones en evaluación

Búsqueda en Kaggle (2026-09-24). Prioridad: **texto rico en español** para lucir chunking + búsqueda semántica.

| Género | Dataset (Kaggle) | Tamaño | Estado |
|--------|------------------|--------|--------|
| 📰 Noticias | Spanish News Classification (Kevin Morgado) | ~1 MB | Candidato fuerte (liviano) |
| 🎬 Cine / reseñas | IMDB 50K Movie Reviews (Spanish) (luisdiegofv97) | ~55 MB | Candidato (recortar para 40 MB) |
| 🏨 Turismo / hoteles | Andalusian Hotels' Reviews | ~5 MB | Candidato |
| 🛒 E-commerce | Amazon Reviews (multilingual, incluye español) | grande | Alternativa |
| 📚 Enciclopédico | Wikibooks / 120M Word Spanish Corpus | 2 GB / 513 MB | Descartado por tamaño |

- **Nota:** el texto médico/enfermedades en español en Kaggle resultó escaso (resultados tabulares o en inglés), por eso se abrió el abanico a otros géneros.
- **Pendiente:** decisión final de temática.

---

## Diario de avances

### 2026-09-24
- Repo clonado y letra del obligatorio subida a `main`.
- Creada rama `develop` para el trabajo.
- Definido stack tecnológico inicial (ver `PLAN.md` §3).
- Creados `PLAN.md` (plan + fases + stack) y `BITACORA.md` (este archivo).
- Exploración de datasets en Kaggle; temática aún sin cerrar.

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
