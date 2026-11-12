"""Ejecutable: genera ``INFORME.pdf`` a partir de ``INFORME.md``.

La letra pide que los documentos de texto vayan en PDF dentro del ZIP. El informe
se escribe en Markdown (se revisa fácil en GitHub y en los diffs) y este script lo
convierte: Markdown -> HTML con ``mistune`` (viene con Jupyter) -> PDF con Chrome
o Edge en modo headless. No hace falta instalar nada aparte.

Uso (desde la raíz del repo, con el venv):
    python -m scripts.informe_a_pdf
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import mistune

from src.config import RAIZ

ENTRADA = RAIZ / "INFORME.md"
SALIDA = RAIZ / "INFORME.pdf"

NAVEGADORES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "google-chrome",
    "chromium",
    "chromium-browser",
    "microsoft-edge",
]

ESTILO = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 20pt; margin: 0 0 6pt; }
h2 { font-size: 14.5pt; margin: 18pt 0 6pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt; break-after: avoid; }
h2.nueva-pagina { break-before: page; margin-top: 0; }
h3 { font-size: 12pt; margin: 12pt 0 4pt; break-after: avoid; }
p, li { margin: 3pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0; font-size: 9.5pt; break-inside: auto; }
tr { break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #f0f0f0; }
code { font-family: Consolas, "Courier New", monospace; font-size: 9pt; background: #f4f4f4; padding: 0 2px; }
pre { background: #f4f4f4; padding: 6pt 8pt; border-radius: 3px; break-inside: avoid; white-space: pre-wrap; }
pre code { background: none; padding: 0; }
hr { border: none; border-top: 1px solid #ccc; margin: 12pt 0; }
a { color: #1a4fa0; }
"""


def buscar_navegador() -> str:
    for candidato in NAVEGADORES:
        ruta = shutil.which(candidato) or (candidato if Path(candidato).exists() else None)
        if ruta:
            return ruta
    raise SystemExit("No se encontró Chrome ni Edge para generar el PDF.")


def main() -> None:
    convertir = mistune.create_markdown(plugins=["table", "strikethrough", "url"])
    cuerpo = convertir(ENTRADA.read_text(encoding="utf-8"))
    # Los anexos empiezan en una página nueva. Se hace acá y no en el Markdown para
    # que INFORME.md se siga viendo limpio en GitHub.
    cuerpo = cuerpo.replace("<h2>Anexo", '<h2 class="nueva-pagina">Anexo')
    html = f'<!doctype html><html lang="es"><head><meta charset="utf-8"><style>{ESTILO}</style></head><body>{cuerpo}</body></html>'

    with tempfile.TemporaryDirectory() as carpeta:
        archivo_html = Path(carpeta) / "informe.html"
        archivo_html.write_text(html, encoding="utf-8")
        subprocess.run(
            [
                buscar_navegador(),
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--user-data-dir={Path(carpeta) / 'perfil'}",
                f"--print-to-pdf={SALIDA}",
                archivo_html.as_uri(),
            ],
            check=True,
            capture_output=True,
        )
    print(f"PDF generado: {SALIDA} ({SALIDA.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
