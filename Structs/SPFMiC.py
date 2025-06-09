import numpy as np
from typing import Any

class SPFMiC:
    #Representação de um Supervised Possibilistic Fuzzy Micro-Cluster (SPFMiC)
    #Armazena as estatísticas de um cluster para atualização incremental

    def __init__(self, centroide: np.ndarray, n_exemplos: int, rotulo: Any, alpha: float, theta: float,
                 tempo: int):
            self.centroide = np.array(centroide)
            self.n = n_exemplos
            self.rotulo = rotulo
            self.alpha = alpha
            self.theta = theta
            self.cf1_pertinencias = np.zeros_like(self.centroide)
            self.cf1_tipicidades = np.zeros_like(self.centroide)
            self.me = 0.0
            self.te = 0.0
            self.ssde = 0.0
            self.t_criacao = tempo
            self.t_atualizacao = tempo

    def __repr__(self):
          return (f"SPFMiC(rotulo = {self.rotulo}, n = {self.n}, centroide = {self.centroide.round(2)})")