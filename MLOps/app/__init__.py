"""
Pacote da API de credit scoring.

Coloca a raiz do repositório no sys.path para que `Model.*` seja importável
tanto rodando local (`uvicorn MLOps.app.api:app`) quanto dentro do container
e nos testes. Antes isso era um hack repetido em api.py e streamlit_app.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
