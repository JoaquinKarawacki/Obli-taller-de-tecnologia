# Guion de la defensa: introducción y demo

Diapositivas 1 a 5. El video nuevo dura unos 5:30 narrado en vivo. Para no pasar los 10 minutos, las diapositivas 1 a 4 tienen que entrar en 1:30 y las 6 a 11 en unos 3:00: hay que recortar los textos de las diapositivas 1 a 4.
**A** y **B** son dos de los tres integrantes; el tercero arranca en la diapositiva 6 con la preparación de los datos.

## Diapositiva 1 · Portada (A, 0:15)

> Buenos días. Somos Nicolás, Pedro y Joaquín, y les vamos a presentar nuestro agente de Agentic RAG sobre correos de phishing y documentos técnicos.

## Diapositiva 2 · Qué hace el agente (A, 0:45)

> El agente trabaja sobre dos fuentes. Por un lado, 82.486 correos de phishing y legítimos, de un dataset público de Kaggle. Por otro, seis documentos largos: los papers de Attention, GPT, GPT-3, Scaling Laws, Subliminal Learning y el libro Designing Data-Intensive Applications.
>
> Ante cada pregunta, el agente decide si puede responder directo o si necesita usar una herramienta para buscar en esas fuentes. Responde en español o en inglés, tiene memoria de corto y de largo plazo, puede pausar y retomar conversaciones, y atiende a varios usuarios en paralelo.

## Diapositiva 3 · El ciclo ReAct (A, 0:45)

> Por dentro es un ciclo ReAct que armamos nosotros con LangGraph: dos nodos, el agente y las herramientas. El usuario entra por consola, por el notebook o por una API con FastAPI. El nodo agente piensa y decide: si necesita información, llama a una herramienta; el resultado vuelve al agente, y el ciclo se repite hasta que puede responder.
>
> Las herramientas usan ChromaDB para la búsqueda semántica y pandas para las estadísticas exactas. La memoria se guarda en SQLite: por conversación y por usuario. El modelo es Qwen 3.8 de 27B, vía Groq; los embeddings, multilingual-e5-small, corren en local.

## Diapositiva 4 · Seis tools y dos memorias (B, 0:45)

> Estas son las seis herramientas que el agente puede elegir. El agente no tiene reglas fijas: elige la herramienta leyendo su descripción, el docstring. Dos buscan en los papers y los listan; tres trabajan sobre el dataset: buscar correos parecidos, contar y armar rankings. La sexta, guardar_recuerdo, guarda datos del usuario.
>
> La memoria tiene dos niveles. La de corto plazo es la conversación: los últimos 12 mensajes, por thread_id. La de largo plazo es del usuario: nombre, trabajo, intereses, por user_id, y sobrevive entre conversaciones. Ahora lo van a ver funcionando.

## Diapositiva 5 · Video de la demo (B, ~5:30)

El video se graba ejecutando `demo_video.ipynb` celda por celda (`abrir_demo.bat`) y recortando las esperas.
Los tiempos de cada tramo son aproximados: se ajustan con el video ya editado.
Si en algún tramo se termina el texto antes de tiempo, se puede quedar callado: la respuesta en pantalla se lee sola.

| Tramo | En pantalla | Qué decir |
|---|---|---|
| ~0:15 | Tabla de secciones y la celda que dice «Agente listo» | «Esta es una notebook de demo; arriba está el mapa de la notebook principal. Primero se carga el agente con sus dos memorias. Debajo de cada respuesta van a ver qué herramienta eligió.» |
| ~0:20 | 1. Small-talk | «Empezamos con un saludo. No hace falta buscar nada, así que responde directo: la línea de tools dice ninguna.» |
| ~0:25 | 2. GPT-3 | «Preguntamos cuántos parámetros tiene GPT-3. Ahora sí eligió buscar_en_papers: recupera fragmentos del paper y responde 175 mil millones, citando el documento y la página.» |
| ~0:25 | 3. «¿Y con cuántos tokens lo entrenaron?» | «Seguimos en la misma conversación. La pregunta no dice GPT-3, dice "lo": lo entiende por el historial. Esa es la memoria de corto plazo.» |
| ~0:25 | 4. Pregunta en inglés | «Ahora en inglés, sobre Scaling Laws. Busca igual y responde en inglés: la pérdida baja como una ley de potencia con el tamaño del modelo.» |
| ~0:30 | 5. Resumen de Attention | «Un pedido de la letra: resumir Attention Is All You Need en tres párrafos. Para resumir trae más fragmentos del mismo paper.» |
| ~0:25 | 6. Correos de phishing | «Una pregunta de conteo. La búsqueda semántica no sirve para contar, así que usa estadisticas_dataset, que cuenta con pandas sobre los 82.486: 42.891 de phishing.» |
| ~0:20 | 7. Ranking de dominios | «También con pandas: los dominios que más phishing mandan, con ranking_dataset.» |
| ~0:30 | 8. Correo sospechoso | «Le pegamos un correo que nos llegó. Usa buscar_en_dataset para traer correos parecidos y explica por qué se parece a phishing: urgencia, cuenta suspendida, pedido de contraseña.» |
| ~0:25 | 9. El mundial 2022 | «Algo que no está en los datos. Ningún fragmento supera el umbral de 0,68 y la herramienta devuelve SIN EVIDENCIA: el agente admite que no tiene información en lugar de inventar.» |
| ~0:30 | 10. «Soy Nico y trabajo en ciberseguridad» | «Esto no es una pregunta: reconoce un dato personal y llama a guardar_recuerdo. Abajo se ve lo que quedó en la memoria de largo plazo.» |
| ~0:25 | 11. «¿Te acordás cómo me llamo?» | «Conversación nueva, sin historial. Igual sabe que soy Nico y trabajo en ciberseguridad, sin usar herramientas: los recuerdos se agregan al prompt en cada turno.» |
| ~0:30 | 12. Reiniciar y retomar | «Reconstruimos el agente desde cero, como si cerráramos el programa, y volvemos a la conversación de GPT-3. Recuerda lo que le preguntamos: el historial queda guardado en SQLite, así que se puede pausar y retomar.» |
| ~0:20 | 13. Otro usuario | «Por último, otro usuario, Ana. No sabe cómo se llama: cada usuario tiene sus propios recuerdos. La memoria no está en el modelo, está en el sistema que lo rodea.» |

Al terminar, pasa la palabra: «Ahora [nombre] les cuenta cómo preparamos los datos para que esto funcione.»
