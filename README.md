# Obli - Taller de Tecnologías 2 — Agentic RAG

Sistema inteligente basado en un **Agentic RAG**: un chatbot en lenguaje natural que responde
consultas sobre un dataset elegido y sobre documentos (papers), usando LLM, embeddings, base
vectorial, agentes con *tool calling* y memoria a corto y largo plazo.

> 📄 Plan completo y fases: [`PLAN.md`](./PLAN.md) · Decisiones y avances: [`BITACORA.md`](./BITACORA.md)

## Stack (100% gratuito)

- **Python 3.11+**
- **LangGraph** — orquestación del agente (grafo armado a mano)
- **Groq** (`llama-3.3-70b-versatile`) — LLM, gratis
- **sentence-transformers** (`multilingual-e5-small`) — embeddings locales, en español
- **ChromaDB** — base vectorial persistente
- **Gradio** — interfaz de chat (demo)

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

# 5. Ejecutar
python main.py
```

## Estructura

```
├── PLAN.md              # plan, fases y stack
├── BITACORA.md          # decisiones de diseño y avances
├── requirements.txt
├── .env.example
├── main.py              # punto de entrada
├── src/                 # módulos del sistema (uno por fase)
│   ├── config.py        # rutas, modelos y constantes
│   ├── datos.py         # Fase 1: consolidación del dataset
│   ├── ingesta.py       # Fase 2: chunking + embeddings + Chroma
│   ├── recuperacion.py  # Fase 3: búsqueda semántica (tool)
│   ├── agente.py        # Fase 4: grafo LangGraph
│   ├── memoria.py       # Fase 5: memoria corto/largo plazo
│   └── chatbot.py       # Fase 6: loop de conversación
├── scripts/
│   └── consolidar_dataset.py
├── data/                # dataset + papers (no se commitea)
└── chroma_db/           # base vectorial persistida (no se commitea)
```

> **Dataset:** Phishing Email Dataset (`naserabdullahalam/phishing-email-dataset`).
> Se descarga de Kaggle (7 CSV) y `scripts/consolidar_dataset.py` los une en un único
> `data/dataset_consolidado.csv` con las columnas `fuente, sender, subject, body, urls, label`.
