"""Punto de entrada del sistema Agentic RAG.

Uso (desde la raíz del repo, con el entorno virtual activado):
    python main.py

Lanza el chatbot por consola (Fase 6), que usa el agente (Fase 4) y las memorias
persistentes (Fase 5). Requiere las bases vectoriales ya construidas (ver README,
scripts construir_indice / construir_indice_papers) y GROQ_API_KEY en .env.
"""
from src import config
from src.chatbot import iniciar_chat


def main():
    if not config.GROQ_API_KEY:
        print("Falta GROQ_API_KEY en .env. Copiá .env.example a .env y completá la clave")
        print("(gratis en https://console.groq.com/keys). Ver README.")
        return
    iniciar_chat()


if __name__ == "__main__":
    main()
