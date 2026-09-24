# PLAN DE TRABAJO — Agentic RAG (Taller de Tecnologías 2)

> Documento maestro: qué hay que hacer, en qué fases y con qué stack.
> Las decisiones de diseño y el avance se registran en [`BITACORA.md`](./BITACORA.md).

- **Materia:** Taller de Tecnologías 2 (7688) — Ingeniería en Sistemas, ORT
- **Entrega:** 16/11/2026 hasta las 21:00 en gestion.ort.edu.uy (ZIP/RAR, máx. 40 MB)
- **Defensa:** oral, presencial, **obligatoria y eliminatoria**
- **Puntaje:** máx. 40 pts / mín. 1 pto
- **Grupo:** hasta 3 personas del mismo dictado

---

## 1. Objetivo

Desarrollar un **Agentic RAG** (chatbot en lenguaje natural) capaz de:

1. Responder consultas sobre el **Phishing Email Dataset** de Kaggle (ciberseguridad, inglés), interactuando en **español o inglés**.
2. Procesar **documentos largos (papers)** vía chunking y responder preguntas sobre ellos.
3. Mantener contexto y orquestar acciones con **agentes + tool calling** y **memoria a corto y largo plazo**.

---

## 2. Requisitos obligatorios (checklist)

### 2.1 Agentic RAG (núcleo)
- [ ] Recibe consultas del usuario.
- [ ] Recupera fragmentos relevantes de la base vectorial con **búsqueda semántica**.
- [ ] Decide cómo combinar la información y generar la respuesta.
- [ ] **Memoria a corto plazo** dentro de la misma conversación.
- [ ] **Múltiples conversaciones independientes**, con pausar/retomar sin pérdida de contexto.
- [ ] **Memoria a largo plazo** compartida entre conversaciones (info del usuario).
- [ ] El sistema **decide qué guardar en memoria y cuándo usarlo** según el contexto.
- [ ] Maneja distintos tipos de consulta: **preguntas sobre documentos / consultas generales / small-talk**, decidiendo dinámicamente.
- [ ] **Evita alucinar**: admite explícitamente cuando no tiene información suficiente.
- [ ] **Múltiples sesiones de usuario en paralelo**, identificadas de forma única.
- [ ] Integra **tool calling** (consulta a BD vectorial, funciones, APIs externas).

### 2.2 Restricciones técnicas
- [ ] Implementado en **Python**.
- [ ] Orquestación con **LangGraph**.
- [ ] ⚠️ **PROHIBIDO** usar `create_agent()`, `create_react_agent()` ni abstracciones de alto nivel que oculten la lógica del agente. **El grafo del agente se construye a mano.**

### 2.3 Chatbot sobre el dataset
- [ ] Responde preguntas sobre la temática elegida (ej.: películas, libros, recetas…).

### 2.4 Conocimiento sobre documentos (papers)
- [ ] Chunking de documentos largos.
- [ ] Embeddings de los fragmentos → base vectorial (Chroma).
- [ ] Recupera los fragmentos más relevantes con búsqueda semántica.
- [ ] Responde preguntas sobre cualquiera de los documentos usando el LLM.

### 2.5 Técnicas comunes
- [ ] Prompt engineering avanzado.
- [ ] Control de alucinaciones.
- [ ] Pipeline RAG completo integrado dentro del agente (recuperación + razonamiento + generación).

### 2.6 Entregables
- [ ] Implementación en **módulos `.py`** (decisión del equipo: más modular y legible que el notebook).
- [ ] **Notebook demo** al cierre que importe los módulos y muestre outputs — para cubrir el "notebook funcional con sus outputs" que menciona la letra (ver nota en `BITACORA.md`).
- [ ] Integra: LLM, embeddings, base vectorial, agentic RAG y chatbot.
- [ ] Documentación completa (ver sección 6).

---

## 3. Stack tecnológico

| Capa | Elección | Notas |
|------|----------|-------|
| Lenguaje | **Python 3.11+** | |
| Entorno | `venv` + `requirements.txt`, `.env` para claves | No commitear claves |
| Orquestación del agente | **LangGraph** (grafo manual) | Sin `create_react_agent` |
| LLM | **Groq** (`llama-3.3-70b-versatile`) | Gratis y rápido, buen tool calling. vía `langchain-groq`. Necesita API key gratuita |
| Embeddings | **`sentence-transformers` multilingüe** (`intfloat/multilingual-e5-small`) | Locales, gratis, buenos para español. Opcional: comparar con otro modelo |
| Base vectorial | **ChromaDB** (persistente en disco) | Colecciones separadas: `dataset` y `papers` |
| Manejo de datos | **pandas** | EDA + carga del dataset |
| Chunking | `langchain-text-splitters` (`RecursiveCharacterTextSplitter`) | Probar tamaños/overlap |
| Lectura de PDFs | **pypdf** / PyMuPDF (`fitz`) | Para los papers |
| Memoria corto plazo | **Checkpointer de LangGraph** (`MemorySaver` → luego `SqliteSaver`) | Aislada por `thread_id` |
| Memoria largo plazo | **Store de LangGraph** (`InMemoryStore` → persistente) | Compartida entre conversaciones, por `user_id` |
| Sesiones paralelas | `thread_id` (conversación) + `user_id` (usuario) | |
| Entregable | **Módulos `.py`** (+ notebook demo opcional al cierre) | Decisión del equipo: más modular y legible |
| Interfaz (demo) | **Gradio** | UI de chat para la defensa (opcional que suma) |
| Otros opcionales | Comparar embeddings/LLMs/chunking | Suman, no restan |

> **Stack 100% gratuito:** Groq (LLM, free tier) + embeddings locales (`sentence-transformers`) + Chroma local. No requiere tarjeta ni servicios pagos.

> El stack puede ajustarse; todo cambio se justifica en `BITACORA.md`.

---

## 4. Fases del trabajo

### Fase 0 — Setup del proyecto
- [ ] Repo + rama `develop`.
- [ ] Entorno virtual + `requirements.txt`.
- [ ] `.env.example` y `.gitignore` (excluir `.env`, `data/`, `chroma_db/`).
- [ ] Estructura de carpetas.

### Fase 1 — Datos
- [ ] Elegir dataset de Kaggle (temática + idioma).
- [ ] Descargar y explorar (EDA con pandas): columnas, tamaño, calidad.
- [ ] Conseguir 1–3 papers para la parte de documentos.

### Fase 2 — Ingesta y embeddings
- [x] Preprocesar dataset (muestra balanceada de 10k por indicación del profe).
- [x] Chunking (RecursiveCharacterTextSplitter, 800/100).
- [x] Generar embeddings (e5 multilingüe) y poblar Chroma (colección `dataset`), indexado por lotes.
- [ ] Poblar la colección `papers` (fase posterior).

### Fase 3 — Retrieval (RAG)
- [ ] Función de búsqueda semántica (top-k) sobre cada colección.
- [ ] Envolver el retrieval como **tool** del agente.
- [ ] Prueba manual de calidad de recuperación.

### Fase 4 — Agente (LangGraph)
- [ ] Definir el estado del grafo.
- [ ] Nodo router: clasifica consulta (documentos / general / small-talk).
- [ ] Nodo(s) de tools (retrieval, otras funciones).
- [ ] Nodo de generación con control de alucinaciones ("no sé si no hay evidencia").
- [ ] Cablear el grafo a mano (nodos + edges condicionales).

### Fase 5 — Memoria y sesiones
- [ ] Memoria corto plazo por `thread_id` (checkpointer).
- [ ] Múltiples conversaciones: pausar/retomar.
- [ ] Memoria largo plazo (store) por `user_id`: decidir qué guardar y cuándo usar.
- [ ] Sesiones de usuario en paralelo con IDs únicos.

### Fase 6 — Chatbot y pruebas
- [ ] Loop de chat en el notebook.
- [ ] Batería de preguntas variadas (dataset, papers, general, small-talk, "no sé").
- [ ] Registrar resultados y ajustar prompts.

### Fase 7 — Documentación y notebook final
- [ ] Completar documentación mínima (sección 6).
- [ ] Notebook limpio, ejecutado end-to-end, con outputs.
- [ ] Citar uso de IA generativa.

### Fase 8 — Opcionales (si da el tiempo)
- [ ] Comparar modelos de embeddings.
- [ ] Comparar estrategias de chunking/recuperación.
- [ ] Comparar LLMs.
- [ ] API (FastAPI) o interfaz (Gradio).

---

## 5. Estructura de carpetas propuesta

```
Obli-taller-de-tecnologia/
├── PLAN.md
├── BITACORA.md
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── main.py                    # punto de entrada
├── src/                       # módulos del sistema (uno por fase)
│   ├── config.py              # rutas, modelos y constantes
│   ├── datos.py               # Fase 1: consolidación del dataset
│   ├── ingesta.py             # Fase 2: chunking + embeddings + Chroma
│   ├── recuperacion.py        # Fase 3: búsqueda semántica (tool)
│   ├── agente.py              # Fase 4: grafo LangGraph
│   ├── memoria.py             # Fase 5: memoria corto/largo plazo
│   └── chatbot.py             # Fase 6: loop de conversación
├── scripts/
│   └── consolidar_dataset.py  # genera data/dataset_consolidado.csv
├── data/                      # dataset + papers (no se commitea)
└── chroma_db/                 # base vectorial persistida (no se commitea)
```

---

## 6. Documentación mínima requerida (para el informe)
- [ ] Arquitectura del agentic RAG y justificación técnica (LLM, embeddings, base vectorial, pipeline, agentes).
- [ ] Estrategia de memoria corto/largo plazo: cómo se almacena, recupera y usa.
- [ ] Ejemplos de prompts y estrategias de prompt engineering.
- [ ] Desafíos encontrados y soluciones.
- [ ] Pruebas y resultados del chatbot.
- [ ] Reflexión sobre mejoras y experimentación opcional.
- [ ] **Citación del uso de IA generativa** (herramientas y contexto de uso).

---

## 7. Presentación en clase
- Presentación grupal oral sobre algún aspecto del obligatorio.
- Entre el **16 y el 27 de noviembre** (tema a coordinar con docentes).

---

## 8. Convenciones del proyecto

### 8.1 Idioma del código
- **Todo el código en español al 100%.** Nombres de variables, funciones, clases, comentarios, docstrings, mensajes de log y prompts se escriben en español.
- Ejemplos: `buscar_fragmentos()`, `memoria_largo_plazo`, `clasificar_consulta()`, `respuesta_generada`.
- Excepción razonable: nombres propios de librerías/APIs (`LangGraph`, `thread_id`, etc.) se mantienen como son.
- Los textos que ve el usuario final del chatbot también en español.

### 8.2 Flujo de trabajo con Git (branching)
- Rama estable de integración: **`develop`**.
- **Una rama por cada tarea/feature**, que sale de `develop`:
  - Nomenclatura: `feature/<descripcion-corta>` (ej.: `feature/ingesta-embeddings`, `feature/agente-router`).
  - Otros prefijos según el caso: `fix/`, `docs/`, `experimento/`.
- Al terminar la tarea, **merge de la rama a `develop`** (y se borra la rama de feature).
- `main` queda como rama de entrega/estable; se mergea `develop → main` en los hitos.
- Commits en español, descriptivos.

```
main ──────●───────────────────●──────  (entregas / hitos)
            \                  /
develop ─────●────●────●──────●────────  (integración)
                   \    \
                    \    feature/agente-router
                     feature/ingesta-embeddings
```

---

## 9. Definiciones y pendientes
- [x] **Dataset / temática:** `naserabdullahalam/phishing-email-dataset` — Phishing Email Dataset (ciberseguridad, **inglés**).
- [x] **Enfoque multilingüe:** datos en inglés, chatbot en **ES + EN** (LLM y embeddings multilingües → retrieval cross-lingual).
- [ ] Integrantes del grupo y números de estudiante.
- [ ] Papers a usar en la parte de documentos.
