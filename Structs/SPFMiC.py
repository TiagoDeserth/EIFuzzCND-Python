import numpy as np
from typing import Any
from .Example import Example

class SPFMiC:
    #Representação de um Supervised Possibilistic Fuzzy Micro-Cluster (SPFMiC)
    #Armazena as estatísticas de um cluster para atualização incremental

    def __init__(self, centroid: np.ndarray, n_examples: int, label: Any, alpha: float, theta: float,
                 time: int):
            self.centroid = np.array(centroid)
            self.n = n_examples
            self.label = label
            self.alpha = alpha
            self.theta = theta
            self.cf1_pertinences = np.zeros_like(self.centroid)
            self.cf1_typicalities = np.zeros_like(self.centroid)
            self.me = 0.0
            self.te = 0.0
            self.ssde = 0.0
            self.t_criation = time
            self.t_atualization = time

    def __repr__(self):
        return (f"SPFMiC(rotulo = {self.label}, n = {self.n}, centroid = {self.centroid.round(2)})")
    
    def assignment_example(self, example: Example, pertinence: float, typicality: float, current_time: int):
        dist_sq = np.sum((example.point - self.centroid) ** 2)

        self.n += 1
        self.me += pertinence ** self.alpha
        self.te += typicality ** self.theta
        self.ssde += pertinence * dist_sq

        self.cf1_pertinences += example.point * pertinence
        self.cf1_typicalities += example.point * typicality

        numerator = (self.alpha * self.cf1_pertinences) + (self.theta * self.cf1_typicalities)
        denominator = (self.alpha * self.me) + (self.theta * self.te)

        if denominator != 0:
             self.centroid = numerator / denominator

        self.t_atualization = current_time

    def calculate_radius(self, scale_factor: float = 2.0) -> float:
        #Calcula o raio de alcance de um Micro-Cluster

        if self.n <= 1: #evita divisão por zero ou raiz de negativo se n=1 e ssde = 0
             return 0.0
        #A fórmula original usa N, não N-1. Manteremos consistência
        return np.sqrt(self.ssde / self.n) * scale_factor
    
    # # * --------------------------------------------------------------------------------------------
    # # * IMPLEMENTAÇÃO DA ATUALIZAÇÃO INCREMENTAL
    # # * Atualiza incrementalmente as estatísticas do SPFMiC com um novo exemplo
    # # * Esta é a implementação que interfere no aprendizado online
    # def assignment_example(self, example, pertinence, typicality, current_time):
    #      self.n_examples += 1
    #      self.time = current_time

    #      # * Atualiza as somas lineares (vetores CF1)
    #      self.cf1_pertinences += pertinence * example.point
    #      self.cf1_typicalities += typicality * example.point

    #      # * Atualiza a soma dos quadrados das distâncias (SSDE)
    #      dis_sq = np.sum((example.point - self.centroid) ** 2)
    #      self.ssde += pertinence * dis_sq

    #      # * Atualiza o M-ésimo momento (Me)
    #      self.me += pertinence ** self.alpha

    #      # * Recalcule o centróide para refletir o novo exemplo
    #      self.centroid = self.cf1_pertinences / self.me