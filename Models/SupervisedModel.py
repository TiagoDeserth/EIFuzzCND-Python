import numpy as np
import pandas as pd
import skfuzzy as fuzz
from typing import Dict, List, Any, Tuple

#Importando as estruturas de dados do nosso próprio pacote
from Structs.Example import Example
from Structs.SPFMiC import SPFMiC

#Funções auxiliares (Distâncias)
def _calculate_euclidian_distance(p1, p2):
    return np.sqrt(np.sum((p1 -p2) ** 2))

def _calculate_pertinence(point, centroid, m):
    return 1.0 / (1.0 + _calculate_euclidian_distance(point, centroid))

class SupervisedModel:
    #Modelo de Classes Conhecidas (MCC)
    #Contém a lógica para o treinamento do modelo na fase Offline

    def __init__(self, fuzziness: float, k_clusters: int, alpha: float, theta: float, min_weight: int):
        self.fuzziness = fuzziness
        self.K = k_clusters
        self.alpha = alpha
        self.theta = theta
        self.min_weight = min_weight
        self.classifier: Dict[Any, List[SPFMiC]] = {} #Estrutura de dicionário que será o MCC | Ele está vazio porque o treinamento ainda não começou | Ele vai armazenar os micro-clusters
        self.known_labels: List[Any] = []

    def classify_new(self, example: Example, current_time: int) -> Tuple[Any, bool]:
        #Tenta Classificar um novo exemplo e atualiza o modelo incrementalmente
        candidates = []
        for spfmics_list in self.classifier.values():
            for spfmic in spfmics_list:
                distance = _calculate_euclidian_distance(example.point, spfmic.centroid)
                if distance <= spfmic.calculate_radius():
                    pertinence = _calculate_pertinence(example.point, spfmic.centroid, self.fuzziness)
                    candidates.append({'spfmic': spfmic, 'pertinence': pertinence, 'typicality': pertinence})

        if not candidates:
            return -1, False
        
        best_candidate = max(candidates, key = lambda c: c['typicality'])
        spfmic_winner = best_candidate['spfmic']
        spfmic_winner.assignment_example(example, 
                                         pertinence = best_candidate['pertinence'], 
                                         typicality = best_candidate['typicality'], 
                                         current_time = current_time)

        return spfmic_winner.label, True

    def _sumarize_clusters_into_spfmics(self, data: np.ndarray,
                                         centroids: np.ndarray,
                                         u_matrix: np.ndarray, 
                                         label: Any) -> List[SPFMiC]:
        
         # Resume os resultados de um clustering em uma lista de SPFMiCs
        spfmics_list = []
        num_clusters = len(centroids)

        #Cria um loop para cada cluster encontrado pelo Fuzzy C-Means
        #A fim de criar um objeto SPFMiC para cada um dos clusters, resumindo suas propriedades
        for i in range(num_clusters):
            centroid_i = centroids[i]

            #Obtém os exemplos que mais pertencem a este cluster
            cluster_membership = u_matrix[i]

            #Encontra quais pontos de dados possuem o maior grau de pertencimento a este cluster específico
            example_indices = np.where(np.argmax(u_matrix, axis = 0) == i)[0]

            if len(example_indices) < self.min_weight:
                continue

            spfmic = SPFMiC(
                centroid = centroid_i,
                n_examples = len(example_indices),
                label = label,
                alpha = self.alpha,
                theta = self.theta,
                time = 0
            )

            ssde = 0.0
            me = 0.0
            cf1_pertinences = np.zeros_like(centroid_i)

            #Itera sobre os membros do cluster e calcula as estatísticas agregadas
            #Aqui é onde se resumo as propriedades do cluster. Em vez de guardar todos os pontos, o SPFMiC guarde essas somas estatísticas. 
            for j in example_indices:
                example_point = data[j]
                pertinence = u_matrix[i, j]
                dis_sq = np.sum((example_point - centroid_i) ** 2)

                ssde += pertinence * dis_sq
                me += pertinence ** self.alpha
                cf1_pertinences += pertinence * example_point

            spfmic.ssde = ssde
            spfmic.me = me
            spfmic.cf1_pertinences = cf1_pertinences
            spfmic.cf1_typicalities = cf1_pertinences

            #Adiciona o micro-cluster recém cirado e preenchido à lista
            spfmics_list.append(spfmic)

        return spfmics_list



