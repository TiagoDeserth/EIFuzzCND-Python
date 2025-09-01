import numpy as np
from typing import List, Any
from Structs.Example import Example
from Structs.SPFMiC import SPFMiC

from .SupervisedModel import _calculate_euclidian_distance, _calculate_pertinence

class NotSupervisedModel:
    #Modelo de Classes Desconhecidas (MCD)

    def __init__(self):
        self.spfmics: List[SPFMiC] = []

    def classify_new(self, example: Example, fuzziness: float) -> Any:
        #Executa a tentativa de classificar um exemplo contra os clusters de novidades existentes

        if not self.spfmics:
            return -1
        
        candidates = []
        for spfmic in self.spfmics:
            distance = _calculate_euclidian_distance(example.point, spfmic.centroid)
            if distance <= spfmic.calculate_radius(scale_factor = 1.5):
                pertinence = _calculate_pertinence(example.point, spfmic.centroid, fuzziness)
                candidates.append({'spfmic': spfmic, 'pertinence': pertinence})

        if not candidates:
            return -1
        
        best_candidate = max(candidates, key = lambda c: c['pertinence'])
        return best_candidate['spfmic'].label
