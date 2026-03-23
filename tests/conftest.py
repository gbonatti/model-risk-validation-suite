"""
conftest.py — Configuração global para a suite de testes (pytest).

Adiciona o diretório src/ ao sys.path para que os testes possam
importar as funções de produção diretamente.
"""
import sys
import os

# Garante que src/ seja encontrado independentemente de onde o pytest é executado
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
