import numpy as np
import pandas as pd
import skfuzzy as fuzz
from typing import Dict, List, Any

#Importando as estruturas de dados do nosso próprio pacote
from Structs.Example import Example
from Structs.SPFMiC import SPFMiC

class SupervisedModel:
    #Modelo de Classes Conhecidas (MCC)
    #Contém a lógica para o treinamento do modelo na fase Offline

    def __init__(self, fuziness: float, k_clusters: int, alpha: float, theta: float, min_weight: int):
        self.fuziness = fuziness
        self.K = k_clusters
        self.alpha = alpha
        self.theta = theta
        self.min_weight = min_weight
        self.classifier: Dict[Any, List[SPFMiC]] = {}
        self.known_labels: List[Any] = []

    def _sumarize_clusters_into_spfmics(self, data: np.ndarray, centroides: np.ndarray,
                                         u_matrix: np.ndarray, label: Any) -> List[SPFMiC]:
        
         #Resume os resultados de um clustering em uma lista de SPFMiCs
        spfmics_list = []
        num_clusters = len(centroides)

        for i in range(num_clusters):
            centroid_i = centroides[i]

            #Obtém os exemplos que mais pertencem a este cluster
            cluster_membership = u_matrix[i]
            example_indices = np.where(np.argmax(u_matrix, axis = 0) == i)[0]

            if len(example_indices) < self.min_weight:
                continue

            spfmic = SPFMiC(
                centroide = centroid_i,
                n_exemplos = len(example_indices),
                rotulo = label,
                alpha = self.alpha,
                theta = self.theta,
                tempo = 0
            )

            ssde = 0.0
            me = 0.0
            cf1_pertinenciais = np.zeros_like(centroid_i)

            for j in example_indices:
                example_point = data[j]
                pertinence = u_matrix[i, j]
                dis_sq = np.sum((example_point - centroid_i) ** 2)

                ssde += pertinence * dis_sq
                me += pertinence ** self.alpha
                cf1_pertinenciais += pertinence * example_point

            spfmic.ssde = ssde
            spfmic.me = me
            spfmic.cf1_pertinenciais = cf1_pertinenciais
            spfmic.cf1_tipicidades = cf1_pertinenciais

            spfmics_list.append(spfmic)

        return spfmics_list



