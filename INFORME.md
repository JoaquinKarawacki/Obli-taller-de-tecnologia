# Agentic RAG sobre correos de phishing y documentos técnicos

**Taller de Tecnologías 2 (7688) — Ingeniería en Sistemas, Universidad ORT Uruguay**

**Obligatorio, setiembre 2026 — Entrega: 16/11/2026**

| Integrante | Número de estudiante |
|---|---|
| Nicolás Castro | 308251 |
| Pedro Morga| 319981  |
| Joaquín Karawacki  | 325535 |

Repositorio: <https://github.com/JoaquinKarawacki/Obli-taller-de-tecnologia>

---

## 1. Resumen

Desarrollamos un chatbot basado en un *agentic RAG* que responde en español o en inglés sobre dos fuentes:

- un **dataset de correos de phishing y legítimos** de Kaggle, de 82.486 correos;
- **seis documentos largos** entregados por la materia: los papers de Attention, GPT, GPT-3, Scaling Laws y Subliminal Learning, y el libro *Designing Data-Intensive Applications*.

Un agente orquestado con LangGraph decide en cada mensaje si responde directamente o si consulta herramientas: búsqueda semántica en una base vectorial, estadísticas del dataset o memoria del usuario. El sistema mantiene memoria a corto plazo por conversación y memoria a largo plazo por usuario, ambas persistidas en SQLite. Permite pausar y retomar conversaciones y atender varias sesiones en paralelo.

El código está en módulos `.py` dentro de `src/`. El notebook `obligatorio_agentic_rag.ipynb` los importa y muestra el sistema funcionando de punta a punta. El mismo agente se puede usar por consola (`main.py`) o por HTTP, con una API hecha con FastAPI.

Desarrollamos los cuatro opcionales de la letra: la comparación de estrategias de chunking y de recuperación, la comparación de modelos de embeddings, la API y la comparación de LLMs. Están en la sección 12 del notebook y se resumen en la sección 7 de este informe.

## 2. Arquitectura

```
Usuario -> chatbot (consola o notebook) o API REST (FastAPI)
              |
              v
   Grafo LangGraph, escrito a mano:

     START -> agente --(¿pidió tools?)--no--> END
                ^            |
                |            sí
                |            v
                +---- herramientas

   Lo que usa el grafo:
     - memoria a corto plazo: SqliteSaver, por thread_id
     - tools de RAG: ChromaDB (búsqueda semántica) y pandas (estadísticas)
     - memoria a largo plazo: SqliteStore, por user_id
```

### 2.1 Componentes

| Capa | Elección | Módulo |
|---|---|---|
| Datos | Phishing Email Dataset, consolidado con pandas | `src/datos.py` |
| Chunking | `RecursiveCharacterTextSplitter`, 800 caracteres con 100 de solapamiento | `src/ingesta.py` |
| Embeddings | `intfloat/multilingual-e5-small`, local | `src/ingesta.py` |
| Base vectorial | ChromaDB persistente, colecciones `dataset` y `papers` | `src/ingesta.py` |
| Recuperación y tools | Búsqueda top-k con umbral, estadísticas y rankings | `src/recuperacion.py` |
| LLM | `qwen/qwen3.8-27b` vía Groq | `src/agente.py` |
| Agente | `StateGraph` con nodos `agente` y `herramientas` | `src/agente.py` |
| Memoria | `SqliteSaver` y `SqliteStore` de LangGraph | `src/memoria.py` |
| Interfaz | Chat por consola con comandos de sesión | `src/chatbot.py`, `main.py` |
| API (opcional) | API REST con FastAPI sobre el mismo agente | `src/api.py` |

### 2.2 Flujo de una consulta

1. El chatbot envía solo el mensaje nuevo, junto con el `thread_id` de la conversación y el `user_id` del usuario.
2. El checkpointer recupera el historial de esa conversación.
3. El nodo `agente` arma el prompt de sistema, agrega los recuerdos del usuario y llama al LLM con las herramientas disponibles.
4. Si el LLM pide herramientas, el nodo `herramientas` las ejecuta y devuelve los resultados al agente, que puede buscar de nuevo o responder.
5. Si el LLM no pide herramientas, la respuesta final vuelve al usuario y el estado queda guardado.

### 2.3 Justificación de las decisiones

**Dataset.** Elegimos un corpus de ciberseguridad con texto real: asunto, cuerpo, remitente y etiqueta. Permite consultas de distinto tipo: ejemplos, patrones, comparación de un correo sospechoso con casos conocidos y conteos. Los datos están en inglés y el chatbot funciona en los dos idiomas.

**Consolidación.** El dataset trae siete CSV. Descartamos `phishing_email.csv`, que ya es una combinación pero pierde el remitente y las URLs, y unimos los otros seis conservando el corpus de origen. Tras quitar duplicados quedan 82.486 correos: 42.891 de phishing y 39.595 legítimos.

**Muestra indexada.** Por indicación docente indexamos una muestra balanceada de 10.000 correos: 5.000 de cada clase, con semilla fija. Las estadísticas, en cambio, se calculan sobre el dataset completo con pandas, porque la búsqueda semántica no sirve para contar.

**Embeddings.** Groq no ofrece embeddings, y elegimos un modelo local para no depender de servicios pagos. `multilingual-e5-small` produce vectores de 384 dimensiones y permite recuperación entre idiomas: una pregunta en español encuentra fragmentos en inglés. El modelo exige los prefijos `query:` y `passage:`, que agrega la clase `EmbeddingsE5`.

**Base vectorial.** ChromaDB funciona en local, sin cuenta ni clave, y persiste en una carpeta. Así cualquier persona puede ejecutar el sistema sin nuestras credenciales y sin depender de la red. Usamos colecciones separadas para correos y documentos, con metadatos para citar y filtrar: título y página en los documentos; corpus, remitente y etiqueta en los correos.

**LLM.** Groq ofrece un plan gratuito y respuestas rápidas. Primero elegimos `llama-3.3-70b-versatile`, pero Groq lo retiró. Comparamos `qwen/qwen3.8-27b` con `openai/gpt-oss-120b` usando nuestras tools: Qwen eligió bien las herramientas y no inventó argumentos, así que lo adoptamos. Después lo confirmamos con una comparación formal de los tres modelos disponibles (sección 7 y sección 12.7 del notebook). Usamos `temperature=0` y sin razonamiento visible, para ahorrar tokens.

**Agente.** La letra prohíbe `create_react_agent()` y otras abstracciones que oculten la lógica. Construimos el grafo a mano y tampoco usamos `ToolNode` ni `tools_condition`, que vienen del mismo módulo `langgraph.prebuilt`. No hay un nodo router aparte: el LLM, guiado por el prompt, decide el tipo de consulta. Un router con otra llamada al LLM duplicaría el consumo de tokens sin agregar capacidades.

### 2.4 Herramientas

| Tool | Función |
|---|---|
| `buscar_en_papers` | Búsqueda semántica en los documentos, con filtro opcional por título aproximado |
| `listar_papers` | Lista de documentos disponibles |
| `buscar_en_dataset` | Búsqueda semántica de correos, con filtro opcional por phishing o legítimo |
| `estadisticas_dataset` | Totales, etiquetas, URLs y desglose por corpus |
| `ranking_dataset` | Remitentes, dominios o asuntos más frecuentes, con la cobertura del cálculo |
| `guardar_recuerdo` | Guarda un dato duradero del usuario en la memoria a largo plazo |

### 2.5 Pipeline RAG

**Ingesta de documentos.** `PyPDFLoader` extrae un documento por página. Normalizamos el texto, uniendo palabras cortadas con guion y espacios repetidos, y lo fragmentamos. Descartamos los fragmentos con menos de 50% de letras, que suelen ser tablas numéricas. Resultado: 2.936 fragmentos.

**Ingesta del dataset.** Unimos asunto y cuerpo de cada correo y lo fragmentamos con el mismo criterio. Resultado: 33.792 fragmentos de los 10.000 correos. La indexación se hace en lotes de 500, porque cargar todo junto agotó la memoria RAM.

**Recuperación.** La búsqueda devuelve los *k* fragmentos más relevantes, 4 por defecto y hasta 10 para resúmenes. Se descartan los que no superan un umbral de relevancia de 0,68. Los fragmentos se reordenan con `LongContextReorder`, que pone los más relevantes en los extremos para mitigar el problema *lost in the middle*. Cada fragmento va numerado y con su fuente, para que la respuesta pueda citarla.

**Generación.** El LLM responde solo con los fragmentos recuperados, cita la fuente y admite cuando no tiene información suficiente.

## 3. Memoria

### 3.1 Corto plazo

- **Cómo se guarda:** el checkpointer `SqliteSaver` de LangGraph guarda el estado de cada conversación, identificada por `thread_id`, después de cada paso del grafo.
- **Cómo se recupera:** al invocar el agente con el mismo `thread_id`, LangGraph carga los mensajes previos. El chatbot solo envía el mensaje nuevo.
- **Cómo se usa:** en cada llamada, el LLM recibe el turno actual completo y una ventana con los últimos 12 mensajes previos. De los turnos anteriores se conservan las preguntas y las respuestas, pero no los fragmentos recuperados. Así se controla el consumo de tokens sin tocar lo guardado.

### 3.2 Pausar y retomar

Como el estado está en un archivo SQLite, una conversación se puede retomar incluso después de cerrar el programa. El chatbot registra cada conversación en un índice por usuario y ofrece:

- `/nueva` — pausa la conversación actual y empieza otra;
- `/conversaciones` — lista las conversaciones guardadas del usuario;
- `/retomar <id>` — vuelve a una conversación con todo su contexto;
- `/usuario <id>` — cambia de usuario.

El índice por usuario también impide retomar conversaciones de otra persona.

### 3.3 Largo plazo

- **Cómo se guarda:** `SqliteStore` guarda los recuerdos en el espacio `("usuarios", user_id, "recuerdos")`. Son compartidos entre todas las conversaciones del usuario.
- **Qué se guarda:** lo decide el agente con la tool `guardar_recuerdo`. El prompt indica guardar datos duraderos sobre la persona: nombre, trabajo, intereses, preferencias y nivel de conocimiento. Indica también no guardar preguntas puntuales ni datos sensibles.
- **Cuándo se usa:** en cada turno, el nodo `agente` lee los recuerdos del usuario y los agrega al prompt de sistema. Son pocos por usuario, así que se incluyen todos sin búsqueda semántica.
- **Aislamiento:** `user_id` y el store se inyectan desde la sesión como `InjectedToolArg`. El LLM no los ve ni los completa, así que no puede escribir en la memoria de otro usuario.

En términos del curso es memoria **semántica**, sobre hechos del usuario, y se escribe **en el camino principal** de la conversación, no en segundo plano.

### 3.4 Sesiones en paralelo

Cada invocación lleva su propio `thread_id` y su `user_id`, por lo que varias sesiones pueden ejecutarse al mismo tiempo sin mezclarse. Probamos 30 sesiones en 8 hilos sobre la misma base SQLite, con un LLM simulado para no consumir cuota, sin errores ni memorias cruzadas. Con el LLM real se probaron dos usuarios simultáneos, en el notebook y en `scripts/probar_memoria.py`. La consola atiende una sesión a la vez.

## 4. Prompt engineering

El prompt de sistema completo está en `src/agente.py` (`PROMPT_SISTEMA`) y se muestra en la sección 6 del notebook.

| Técnica | Aplicación |
|---|---|
| Rol | "Asistente experto en ciberseguridad (phishing) y en los papers de la materia" |
| Instrucciones por tipo de consulta | Small-talk sin tools; documentos y dataset siempre con tools; conocimiento general con una nota aclaratoria |
| Descripción de herramientas | El prompt y los docstrings explican cuándo conviene cada tool; el LLM lee los docstrings para elegir |
| Control de alucinaciones | Responder solo con los fragmentos; no agregar calificativos no respaldados; admitir falta de evidencia; citar la fuente |
| Few-shot | Tres ejemplos: dato presente en el fragmento, dato ausente y consulta general |
| Idioma espejo | Responder en el idioma del usuario aunque los documentos estén en inglés |
| Formato y longitud | Respuestas concisas, con un máximo de 400 palabras |
| Memoria guiada | Qué guardar, qué no guardar y cómo usar los recuerdos |

Ejemplo few-shot incluido en el prompt:

```
Usuario: ¿Cuánto costó entrenar el modelo del paper X?
Fragmento recuperado: "[1] Fuente: X, pág. 7. We used the Adam optimizer."
Respuesta: No tengo información suficiente en los documentos: los fragmentos
recuperados no mencionan el costo de entrenamiento.
```

Agregamos los ejemplos después de detectar en las pruebas dos fallas. El agente describió un modelo de GPT-3 como "el más pequeño de la familia", algo que el paper no dice, y respondió una consulta general sin aclarar que no venía de los documentos.

### 4.1 Control de alucinaciones

Aplicamos varias capas:

1. **Umbral de relevancia:** si ningún fragmento supera 0,68, la tool devuelve `SIN EVIDENCIA` en lugar de texto irrelevante.
2. **Filtro de ingesta:** se descartan fragmentos casi sin texto, como tablas, que aparecían como resultado de cualquier consulta.
3. **Errores explícitos:** si una fuente no está disponible, la tool devuelve `ERROR` en lugar de "no hay resultados".
4. **Prompt:** responder solo con evidencia, citar la fuente y admitir cuando falta información.
5. **Búsqueda en seguimientos:** toda pregunta de seguimiento que pida un dato nuevo exige volver a buscar.
6. **Control de acciones:** si la respuesta dice que guardó un dato del usuario pero en ese turno no se llamó a `guardar_recuerdo`, el nodo `agente` se lo indica al LLM y pide la respuesta de nuevo, una sola vez. Evita que el sistema afirme haber hecho algo que no hizo.

## 5. Desafíos y soluciones

| Desafío | Solución |
|---|---|
| Los puntajes de e5 se solapan: consultas pertinentes de 0,72 a 0,83 y ajenas de 0,63 a 0,75 | El umbral es solo un piso; el resto del filtrado lo hace el prompt |
| Las tablas numéricas aparecían como resultado de consultas sin relación | Filtro de fragmentos con menos de 50% de letras |
| Algunas consultas ajenas igual superan el umbral; por ejemplo, "receta de lasaña" con 0,72 | Documentado como limitación; lo cubre el prompt |
| Construir el índice de una sola vez agotó la RAM | Indexación en lotes de 500 fragmentos |
| Groq retiró Llama 3.3 | Comparación y cambio a Qwen 3.8 27B |
| Con el plan gratuito, Groq rechazaba pedidos con un error 429 que los reintentos no resolvían | `max_tokens=900`, reintentos con espera y un aviso claro cuando se agota la cuota diaria |
| Una tool que fallaba dejaba la conversación inutilizable | El nodo de herramientas siempre devuelve una respuesta, con el error como texto |
| El chat no permitía retomar conversaciones guardadas | Índice por usuario y comandos `/conversaciones` y `/retomar` |
| Nombres de papers como "gpt3" o "transformer" no se encontraban | Patrones de nombres habituales; ante ambigüedad se lista lo disponible |
| El historial crecía con los fragmentos recuperados | Ventana deslizante sin los fragmentos de turnos anteriores |
| Con ese recorte, una pregunta de seguimiento inventó datos | El prompt exige buscar todo dato nuevo en los seguimientos |
| El modelo a veces armaba mal una tool o filtraba su razonamiento | Reintento y limpieza de la respuesta |
| `database is locked` al usar checkpointer y store sobre el mismo archivo | Conexiones SQLite en modo autocommit |
| El agente decía "lo guardé" sin llamar a `guardar_recuerdo`, y el dato se perdía | Regla y ejemplo en el prompt, y un control en el código que detecta esa afirmación y pide la respuesta de nuevo |

## 6. Pruebas y resultados

Los resultados de esta sección son los del notebook ejecutado el 12/11/2026 con el LLM real (sección 8 y siguientes), con la versión final del prompt y del código.

### 6.1 Decisión del tipo de consulta

Probamos 12 consultas de distinto tipo. Para cada una declaramos de antemano qué tools debía usar el agente.

| Tipo | Consulta | Tools usadas | Correcto | Tiempo |
|---|---|---|---|---|
| Small-talk | Hola! cómo andás? | ninguna | Sí | 1 s |
| General | ¿Cuál es la capital de Francia? | ninguna | Sí | 1 s |
| Documentos (ES) | ¿Cuántos parámetros tiene GPT-3? | `buscar_en_papers` | Sí | 21 s |
| Documentos (EN) | What does the Scaling Laws paper say about model size and loss? | `buscar_en_papers` | Sí | 51 s |
| Resumen | Resumir "Attention Is All You Need" en tres párrafos | `buscar_en_papers` | Sí | 69 s |
| Sin evidencia | ¿Qué dice el paper de Attention sobre el mundial 2022? | `buscar_en_papers` | Sí | 35 s |
| Estadística | ¿Cuántos correos de phishing hay en el dataset? | `estadisticas_dataset` | Sí | 38 s |
| Ranking | ¿Qué dominios usan más los correos de phishing? | `ranking_dataset` | Sí | 37 s |
| Dataset | Mostrame un ejemplo de correo de phishing bancario | `buscar_en_dataset` | Sí | 42 s |
| Análisis | 'Your account is suspended, click here'. ¿Es phishing? | `buscar_en_dataset` | Sí | 41 s |
| Contexto | ¿Qué es multi-head attention? | `buscar_en_papers` | Sí | 43 s |
| Seguimiento | ¿Y cuántas cabezas usa el modelo big? | `buscar_en_papers` | Sí | 49 s |

**Resultado: 12 de 12 decisiones correctas, con 36 segundos promedio por consulta.** Casi todo ese tiempo es espera por el límite de tokens por minuto de Groq. Es el mismo resultado que en la ejecución anterior, del 9/11, hecha antes de los cambios de prompt del 10/11: esos cambios no empeoraron la elección de herramientas.

### 6.2 Calidad de las respuestas

Verificamos contra los documentos los datos de las respuestas:

- **Cifras correctas:** GPT-3 tiene 175B de parámetros; el Transformer obtiene 28.4 BLEU en inglés-alemán y 41.8 en inglés-francés; el modelo base usa 8 cabezas con `d_k = d_v = 64`, y el modelo big, 16 cabezas con `d_model = 1024`.
- **Seguimiento:** "¿Y cuántas cabezas usa el modelo big?" se resolvió buscando de nuevo y citando la Tabla 3. En una prueba anterior a la corrección del prompt, la misma pregunta se respondió sin buscar y con datos falsos.
- **Seguimiento sin buscar, cuando corresponde:** por la API, "¿Y qué valores de beta usaron?" se respondió sin tools, porque los valores ya estaban escritos en la respuesta anterior. Es lo que permite el prompt: reutilizar lo que aparece en respuestas previas y buscar todo dato nuevo.
- **Sin evidencia:** ante la consulta sobre el mundial, el agente buscó y admitió que el paper no trata ese tema.
- **Con y sin contexto:** sin contexto, el LLM inventó una explicación del *subliminal learning* como si fuera destilación de conocimiento. Con los fragmentos respondió lo que dice el paper: el maestro y el estudiante deben compartir una inicialización similar.
- **Estadísticas:** los conteos coinciden con el CSV. El ranking informa sobre cuántos correos se calculó: 27.139 de 42.891.

### 6.3 Memoria y sesiones

| Prueba | Resultado |
|---|---|
| Ana se presenta, sin pedir que se recuerde | El agente guardó el recuerdo "Se llama Ana y trabaja en seguridad informática en un banco" |
| Ana pregunta qué es el *few-shot learning* | Buscó en los documentos y citó el paper de GPT-3 (pág. 6); no guardó la pregunta como recuerdo |
| Conversación nueva de Ana | Recordó su nombre y trabajo (largo plazo), pero no lo que preguntó en la otra conversación |
| Reinicio y retomar la conversación 1 | Con un agente nuevo, recordó que la última pregunta fue sobre few-shot learning |
| Ana y Beto en paralelo | Cada uno recibió solo su información; Beto no tiene recuerdos |
| Chat con `/nueva`, `/conversaciones` y `/retomar` | Listó las dos conversaciones y retomó la primera |

### 6.4 Problemas observados

- **Pregunta sin búsqueda (no se repitió):** en la ejecución del 9/11, "¿Qué es el few-shot learning?" se respondió con conocimiento general y la nota aclaratoria, aunque el tema está en el paper de GPT-3. En la del 12/11, con el prompt final, el agente buscó y citó el paper.
- **Búsqueda sin reintento:** al pedir un ejemplo de phishing bancario, el agente dijo que no encontró uno específico, mostró cuatro correos de phishing de otro tipo y ofreció buscar de nuevo, en lugar de reformular la búsqueda por su cuenta. Pasó en las dos ejecuciones. En otra prueba, con otra consulta, encontró un correo que suplanta a Bank of America. La respuesta no inventa un ejemplo, pero el prompt ya le permite buscar de nuevo y no lo hizo.
- **Interpretaciones de más:** en el ranking de dominios, el agente agregó que los atacantes usan servicios de correo gratuitos para suplantar identidades. Es una explicación razonable, pero no surge de los datos. Ante la consulta sin evidencia, describió el paper de Attention como enfocado en "modelos de lenguaje", cuando sus experimentos son de traducción automática. Son agregados menores, fuera del dato pedido, que la regla de no agregar calificativos no termina de evitar.
- **Recuerdo no guardado (corregido):** en pruebas posteriores, ante "Me llamo Ana y trabajo en un banco, recordalo", el agente respondía "Guardé que trabajás en un banco" sin llamar a `guardar_recuerdo`, y el dato no quedaba guardado. Pasaba siempre con ese mensaje, por `temperature=0`. Tampoco guardaba "Soy desarrollador y me interesa la seguridad informática". Se corrigió en dos capas: el prompt ahora exige guardar antes de responder y prohíbe afirmar un guardado sin la tool, con un ejemplo few-shot; y un control en el nodo `agente` detecta la afirmación sin llamada y pide la respuesta de nuevo. Después de la corrección, los cinco mensajes de prueba guardaron su recuerdo y el pedido de guardar una contraseña se rechazó. En una de las pruebas fue el control el que resolvió el caso. En la ejecución del 12/11, Ana y Sofía guardaron su recuerdo al presentarse.
- **Demo de retomar:** en el chat guionado, la pregunta "¿Cómo me llamo?" también podía responderse con la memoria a largo plazo. La prueba concluyente de la memoria a corto plazo es la del reinicio, en la tabla anterior.

## 7. Limitaciones y mejoras posibles

**Limitaciones:**

- **Cuota de Groq:** el plan gratuito permite 200.000 tokens por día y 8.000 por minuto. Una pregunta con búsqueda hace al menos dos llamadas al LLM, por lo que el sistema admite unas 40 preguntas por día y algunas respuestas tardan decenas de segundos.
- **Resúmenes:** se generan con hasta 10 fragmentos, no con el documento completo.
- **Páginas:** las citas usan el número de página del PDF, que puede no coincidir con el número impreso, sobre todo en el libro.
- **Etiquetas del dataset:** lo marcado como phishing también incluye spam común, y el remitente falta en cerca del 40% de los correos.
- **Control de alucinaciones:** reduce el riesgo, pero depende del prompt y no lo elimina.
- **Idioma al guardar recuerdos:** si el usuario cuenta un dato personal en inglés, el agente lo guarda pero a veces confirma en español.

**Mejoras posibles:**

- Escribir en inglés la consulta de `buscar_en_papers`: en los experimentos de la sección 12, las mismas preguntas recuperan la página correcta en 70% de los casos en inglés y en 45% en español.
- Completar siempre `titulo_paper` cuando la pregunta nombra un paper: buscar dentro del paper correcto fue lo que más mejoró la recuperación.
- Pasar a `multilingual-e5-base` y recalibrar el umbral, si se puede pagar una reindexación entre dos y casi cuatro veces más lenta.
- Verificar cada respuesta contra los fragmentos con un nodo de validación, cuando la cuota lo permita.
- Reformular automáticamente la búsqueda cuando los fragmentos no responden la pregunta.
- Re-ranking de los fragmentos recuperados y búsqueda híbrida (semántica y por palabras clave).
- Resúmenes jerárquicos por sección para documentos largos.
- Actualizar y borrar recuerdos, y agregar memoria episódica.
- Interfaz web (por ejemplo con Gradio) sobre la API.

**Opcionales de la letra (sección 12 del notebook):**

Se desarrollaron los cuatro opcionales. Los dos que corresponden a temas vistos en clase son las estrategias de chunking y de recuperación y la comparación de modelos de embeddings; se evaluaron sin el LLM, con 20 preguntas sobre los cinco papers en español y en inglés, midiendo si se recupera la página que contiene la respuesta (acierto@4 y MRR).

- **Chunking:** se compararon el corte recursivo con cuatro tamaños, sin solapamiento, por oraciones y semántico. Ninguna estrategia mejoró de forma clara (diferencias de una a tres preguntas), así que se mantiene 800/100.
- **Recuperación:** buscar dentro del paper correcto es lo que más mejora (de 45% a 60% en español y de 70% a 85% en inglés), lo que respalda el filtro `titulo_paper`. MMR no fue una mejora estable y top-8 duplica los tokens. El umbral 0,68 no perdió preguntas pertinentes.
- **Embeddings:** se compararon `all-MiniLM-L6-v2` (el de los labs), `paraphrase-multilingual-MiniLM-L12-v2`, `multilingual-e5-small` y `multilingual-e5-base`. El de los labs acierta 30% en español. e5-base es el mejor (65% en español contra 45% de e5-small), pero indexa entre dos y casi cuatro veces más lento, según la ejecución; se mantiene e5-small por costo. Las métricas de calidad dieron idénticas en las dos ejecuciones del notebook; solo cambiaron los tiempos.
- **Hallazgo principal:** las preguntas en español recuperan peor que las mismas en inglés. Escribir la consulta de la tool en inglés es una mejora barata que queda propuesta, porque hay que validarla con el LLM.

- **API con FastAPI** (`src/api.py`): expone el mismo agente por HTTP, sin duplicar lógica. Permite conversar (`POST /chat`, que crea o continúa una conversación por `thread_id`), listar conversaciones, leer el historial desde el checkpointer y consultar los recuerdos del usuario. El agente se construye una vez al arrancar; los endpoints corren en un pool de hilos para atender usuarios en paralelo; un usuario no puede leer ni continuar conversaciones de otro (404); y los errores se traducen a códigos HTTP (422 entrada inválida, 429 cuota de Groq agotada, 502 otros errores del LLM). FastAPI genera la documentación interactiva en `/docs`. Se levanta con `uvicorn src.api:app`.

- **Comparación de LLMs** (`src/comparacion_llms.py`): el mismo agente, cambiando solo el LLM, sobre 10 casos de la batería, con `qwen/qwen3.8-27b`, `openai/gpt-oss-120b` y `openai/gpt-oss-20b`. Los tres eligieron bien las tools en los 10 casos, así que las diferencias están en la respuesta. En la revisión manual, Qwen no tuvo observaciones (10 de 10). gpt-oss-120b, el más rápido (20,7 s de media contra 30,7), perdió el contexto en el seguimiento: interpretó "modelo big" como GPT-3 y respondió con un dato real de otro paper. Los dos gpt-oss respondieron en español a una pregunta en inglés, y gpt-oss-20b presentó como bancario un correo que no lo era. El costo fue parecido, unos 40.000 tokens de entrada por batería. Se mantiene Qwen. Los resultados quedan guardados en `resultados/comparacion_llms.json`, para que el notebook los muestre sin gastar cuota.

## 8. Uso de inteligencia artificial generativa

- **Herramienta:** Claude Code (Anthropic).
- **Contexto de uso:** estructura del proyecto, redacción de documentación, exploración de datasets, análisis del estado del proyecto, corrección de errores, implementación de mejoras del chatbot, ejecución de pruebas, armado y ejecución del notebook, revisión del proyecto contra la letra, redacción inicial de este informe, su actualización con los resultados del 12/11 y el script que lo convierte a PDF (`scripts/informe_a_pdf.py`).
- **Revisión:** [PENDIENTE: cada integrante debe confirmar que esta lista refleja todo el uso real de IA y que revisó el contenido.]

Todo el contenido generado con IA fue revisado y verificado por el grupo, que es responsable por los errores que pueda contener.

## Anexo: ejecución

Los pasos completos están en `README.md`. En resumen:

```bash
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env    # y completar GROQ_API_KEY
python -m scripts.consolidar_dataset
python -m scripts.construir_indice
python -m scripts.construir_indice_papers
python main.py            # chatbot por consola
uvicorn src.api:app       # o la API REST, con documentación en /docs
```

En esta máquina, construir el índice de los documentos tardó unos 4 minutos y el del dataset, unos 30.
