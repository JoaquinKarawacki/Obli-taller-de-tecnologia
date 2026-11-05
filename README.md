# Obli - Taller de Tecnologías 2 — Agentic RAG

Sistema inteligente basado en un **Agentic RAG**: un chatbot en lenguaje natural que responde
consultas sobre un dataset elegido y sobre documentos (papers), usando LLM, embeddings, base
vectorial, agentes con *tool calling* y memoria a corto y largo plazo.

> 📄 Plan completo y fases: [`PLAN.md`](./PLAN.md) · Decisiones y avances: [`BITACORA.md`](./BITACORA.md)

## Estado actual (5/11/2026): trabajo en curso

**Todavía no está listo para entregar.** Entrega: 16/11/2026 a las 21:00.

Falta terminar:

- [ ] **Notebook:** `obligatorio_agentic_rag.ipynb` está armado, pero **no tiene outputs**. Las secciones 1 a 6 se ejecutaron con datos reales; las secciones 7 a 10 solo se probaron con un LLM simulado. Falta ejecutarlo con Groq.
- [ ] **Informe en PDF:** no está empezado. El material base está en `BITACORA.md`.
- [ ] **Prueba final con Groq:** el reintento de tools mal armadas, la limpieza de `</think>` y el aviso de cuota agotada solo se probaron de forma simulada.
- [ ] **`requirements.txt`:** las versiones no están fijadas.
- [ ] **ZIP de entrega:** el CSV y `chroma_db/` superan los 40 MB. Hay que definir con los docentes cómo se evalúa.
- [ ] **Integración:** estos cambios están en `feature/calidad-respuestas`; falta mergearlos a `develop` y después a `main`.

A confirmar con los docentes: la letra dice hasta 3 integrantes en la página 1 y hasta 2 en la página 4.

**Cuota de Groq:** el plan gratuito permite 200.000 tokens por día y 8.000 por minuto, lo que da unas 40 preguntas por día. Conviene no gastarla en pruebas antes de ejecutar el notebook o defender.

## Stack (100% gratuito)

- **Python 3.11+**
- **LangGraph** — orquestación del agente (grafo armado a mano)
- **Groq** (`qwen/qwen3.8-27b`) — LLM, gratis
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

## Estructura

```
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
│   └── chatbot.py       # Fase 6: loop de conversación
├── scripts/
│   ├── consolidar_dataset.py
│   ├── construir_indice.py
│   ├── construir_indice_papers.py
│   └── probar_recuperacion.py
├── data/                # dataset + papers (no se commitea)
└── chroma_db/           # base vectorial persistida (no se commitea)
```

> **Dataset:** Phishing Email Dataset (`naserabdullahalam/phishing-email-dataset`).
> Se descarga de Kaggle (7 CSV) y `scripts/consolidar_dataset.py` los une en un único
> `data/dataset_consolidado.csv` con las columnas `fuente, sender, subject, body, urls, label`.
