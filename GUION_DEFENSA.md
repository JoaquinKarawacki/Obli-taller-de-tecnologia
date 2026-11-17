# Guion de la defensa: introducción y demo

Diapositivas 1 a 5. Dura unos 4:45: 2:30 de introducción y 2:15 de video narrado en vivo.
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

## Diapositiva 5 · Video de la demo (B, 2:15)

Los tiempos son del video ya recortado: empieza en el 0:55 del original y termina en el 3:06.
Si en algún tramo se termina el texto antes de tiempo, se puede quedar callado: la respuesta en pantalla se lee sola.

| Video | En pantalla | Qué decir |
|---|---|---|
| 0:00–0:15 | Tabla de secciones y la celda que dice «Agente listo» | «Esta es una notebook de demo; arriba está el mapa de la notebook principal, con sus doce secciones. Primero se carga el agente con sus dos memorias. Cada pregunta va en una conversación nueva.» |
| 0:15–0:35 | Pregunta 1: GPT-3 | «Preguntamos cuántos parámetros tiene GPT-3. Fíjense en la línea de tools: eligió buscar_en_papers. Recupera fragmentos del paper y responde 175 mil millones, citando el documento y la página.» |
| 0:35–0:50 | Pregunta 2: correos de phishing | «Ahora una pregunta de conteo. Acá la búsqueda semántica no sirve, porque trae solo algunos correos parecidos. Entonces usa estadisticas_dataset, que cuenta con pandas sobre los 82.486: son 42.891 de phishing, con el desglose por fuente.» |
| 0:50–1:15 | Pregunta 3: el mundial 2022 (la espera más larga) | «Ahora algo que no está en los datos: qué dice el paper de Attention sobre el mundial 2022. Mientras espera, les cuento qué pasa: busca en el paper, pero ningún fragmento supera el umbral de similitud de 0,68, y la herramienta le devuelve SIN EVIDENCIA. Por el prompt, en ese caso tiene que admitirlo. Y efectivamente responde que no tiene información, en lugar de inventar.» |
| 1:15–1:50 | Pregunta 4: «Soy Nico y trabajo en ciberseguridad» | «Ahora le contamos quiénes somos. Esto no es una pregunta: el agente reconoce que es un dato personal y llama a guardar_recuerdo. Eso lo escribe en la memoria de largo plazo, asociada a este usuario. Abajo se ve lo que quedó guardado: se llama Nico y trabaja en ciberseguridad.» |
| 1:50–2:15 | Pregunta 5: «¿Te acordás cómo me llamo?» | «Y la prueba: una conversación nueva, con otro thread_id y sin historial. Igual responde que me llamo Nico y trabajo en ciberseguridad. No usó ninguna herramienta, porque los recuerdos del usuario se agregan al prompt en cada turno. La memoria no está en el modelo, está en el sistema que lo rodea.» |

Al terminar, pasa la palabra: «Ahora [nombre] les cuenta cómo preparamos los datos para que esto funcione.»
