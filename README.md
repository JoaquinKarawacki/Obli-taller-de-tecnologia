# Obli - Taller de Tecnologías 2 — Agentic RAG

Sistema inteligente basado en un **Agentic RAG**: un chatbot en lenguaje natural que responde
consultas sobre un dataset elegido y sobre documentos (papers), usando LLM, embeddings, base
vectorial, agentes con *tool calling* y memoria a corto y largo plazo.

> 📄 Plan completo y fases: [`PLAN.md`](./PLAN.md) · Decisiones y avances: [`BITACORA.md`](./BITACORA.md)

## Estado actual (12/11/2026): trabajo en curso

**Todavía no está listo para entregar.** Entrega: 16/11/2026 a las 21:00.

Hecho:

- [x] **Código:** todo integrado en `develop`, incluidos los opcionales.
- [x] **Notebook:** `obligatorio_agentic_rag.ipynb`, ejecutado completo con Groq y con outputs el 12/11, con el prompt y el código finales. 12 de 12 decisiones de tools correctas.
- [x] **Informe:** completo en `INFORME.md`, con los resultados del 12/11, salvo la revisión del uso de IA. `python -m scripts.informe_a_pdf` genera `INFORME.pdf`, que no se commitea.
- [x] **`requirements.txt`:** versiones fijadas, probadas con Python 3.14.3.
- [x] **Opcionales:** los cuatro de la letra, en la sección 12 del notebook: experimentos de chunking, recuperación y embeddings, una API con FastAPI (`src/api.py`) y la comparación de LLMs (`src/comparacion_llms.py`).

Falta terminar:

- [ ] **Informe:** completar el `[PENDIENTE]` de la sección 8 (revisión del uso de IA), revisarlo entre los tres y regenerar el PDF. La letra pide que los documentos de texto vayan en PDF dentro del ZIP.
- [ ] **Prueba con Groq:** el reintento de tools mal armadas, la limpieza de `</think>` y el aviso de cuota agotada solo se probaron de forma simulada, porque esas fallas no se dieron en la ejecución real.
- [ ] **Instalación limpia:** las versiones fijadas no se probaron desde un entorno nuevo.
- [ ] **ZIP de entrega:** máximo 40 MB. `data/` (431 MB) y `chroma_db/` (221 MB) no entran: el ZIP lleva el código, el notebook con outputs, el informe en PDF y el link al repositorio. Confirmar con los docentes.
- [ ] **Integración:** mergear `develop` a `main` cuando esté todo cerrado. Hoy `main` solo tiene la letra.
- [ ] **Defensa y presentación oral:** preparar entre el 16 y el 27 de noviembre.

A confirmar con los docentes: la letra dice hasta 3 integrantes en la página 1 y hasta 2 en la página 4.

**Cuota de Groq:** el plan gratuito permite 200.000 tokens por día y 8.000 por minuto, lo que da unas 40 preguntas por día. Conviene no gastarla en pruebas antes de ejecutar el notebook o defender.

## Stack (100% gratuito)

- **Python 3.11+** (las versiones de `requirements.txt` se probaron con 3.14.3)
- **LangGraph** — orquestación del agente (grafo armado a mano)
- **Groq** (`qwen/qwen3.8-27b`) — LLM, gratis
- **sentence-transformers** (`multilingual-e5-small`) — embeddings locales, en español
- **ChromaDB** — base vectorial persistente
- **FastAPI** — API REST del agente (opcional)

## Convenciones

- **Código 100% en español** (variables, funciones, comentarios, prompts).
- **Flujo Git:** una rama por tarea (`feature/...`) que sale de `develop` y se mergea de vuelta.
  `main` es la rama estable de entrega.

## Puesta en marcha

```bash
# 1. Crear y activar entorno virtual
python -m venv venv
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar la API key de Groq (gratis: https://console.groq.com/keys)
cp .env.example .env      # en Windows: copy .env.example .env
# editar .env y completar GROQ_API_KEY=...

# 4. Dataset: dejar los CSV de Kaggle en data/ y consolidarlos en uno solo
python -m scripts.consolidar_dataset   # genera data/dataset_consolidado.csv

# 5. Construir los índices vectoriales (ChromaDB, carpeta chroma_db/)
python -m scripts.construir_indice          # dataset (~10k correos)
# copiar los PDFs de "Obligatorio setiembre 2026 - Papers" a data/papers/
python -m scripts.construir_indice_papers   # papers (~9 min en CPU)
python -m scripts.probar_recuperacion       # (opcional) probar búsqueda y umbral

# 6. Ejecutar el chatbot (requiere GROQ_API_KEY en .env)
python main.py
```

En el chat se puede preguntar en español o inglés sobre los papers y el dataset.
Comandos disponibles:

- `/nueva` — pausar la conversación actual y empezar una nueva
- `/conversaciones` — listar las conversaciones guardadas del usuario
- `/retomar <id>` — volver a una conversación guardada, incluso tras reiniciar el programa
- `/usuario <id>` — cambiar de usuario (otros recuerdos de largo plazo) y empezar conversación nueva
- `/ayuda` — ver la ayuda
- `/salir` — terminar

### API REST (opcional)

El mismo agente también se puede usar por HTTP:

```bash
uvicorn src.api:app
```

La documentación interactiva queda en http://127.0.0.1:8000/docs. Endpoints: `GET /salud`,
`POST /chat` (con `user_id`, `mensaje` y, para continuar una conversación, `thread_id`),
`GET /usuarios/{user_id}/conversaciones`, `GET /usuarios/{user_id}/conversaciones/{thread_id}` y
`GET /usuarios/{user_id}/recuerdos`.

## Estructura

```
├── INFORME.md           # informe de entrega (documentación de la letra)
├── PLAN.md              # plan, fases y stack
├── BITACORA.md          # decisiones de diseño y avances
├── requirements.txt
├── .env.example
├── main.py              # punto de entrada
├── obligatorio_agentic_rag.ipynb  # notebook entregable, con el sistema de punta a punta
├── src/                 # módulos del sistema (uno por fase)
│   ├── config.py        # rutas, modelos y constantes
│   ├── datos.py         # Fase 1: consolidación del dataset
│   ├── ingesta.py       # Fase 2: chunking + embeddings + Chroma
│   ├── recuperacion.py  # Fase 3: búsqueda semántica (tool)
│   ├── agente.py        # Fase 4: grafo LangGraph
│   ├── memoria.py       # Fase 5: memoria corto/largo plazo
│   ├── chatbot.py       # Fase 6: loop de conversación
│   ├── api.py           # opcional: API REST con FastAPI
│   ├── comparacion_llms.py  # opcional: comparación de LLMs
│   └── experimentos.py  # opcionales: chunking, recuperación y embeddings
├── resultados/          # resultados guardados de la comparación de LLMs
├── scripts/
│   ├── consolidar_dataset.py
│   ├── construir_indice.py
│   ├── construir_indice_papers.py
│   ├── probar_recuperacion.py
│   ├── probar_agente.py     # batería de consultas (usa Groq)
│   ├── probar_memoria.py    # memoria y sesiones (usa Groq)
│   └── informe_a_pdf.py     # INFORME.md -> INFORME.pdf (Chrome o Edge headless)
├── data/                # dataset + papers (no se commitea)
└── chroma_db/           # base vectorial persistida (no se commitea)
```

> **Dataset:** Phishing Email Dataset (`naserabdullahalam/phishing-email-dataset`).
> Se descarga de Kaggle (7 CSV) y `scripts/consolidar_dataset.py` los une en un único
> `data/dataset_consolidado.csv` con las columnas `fuente, sender, subject, body, urls, label`.
