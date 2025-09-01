import numpy as np
from typing import Any

class Example:
    #Representa uma única instância de dados
    #Encapsula os atributos (ponto) e o rótulo da classe

    def __init__(self, point: np.ndarray, true_label: Any, time: int = 0):
        self.point = point
        self.true_label = true_label
        self.classified_label = None
        self.time = time

    def __repr__(self):
        return (f"Example(true_label = {self.true_label}, ponto = {self.point})")