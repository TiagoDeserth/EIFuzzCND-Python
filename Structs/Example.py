import numpy as np
from typing import Any

class Example:
    #Representa uma única instância de dados
    #Encapsula os atributos (ponto) e o rótulo da classe

    def __init__(self, ponto: np.ndarray, rotulo_verdadeiro: Any, tempo: int = 0):
        self.ponto = ponto
        self.rotulo_verdadeiro = rotulo_verdadeiro
        self.rotulo_classificado = None
        self.tempo = tempo

    def __repr__(self):
        return (f"Example(rotulo_verdadeiro = {self.rotulo_verdadeiro}, ponto = {self.ponto})")