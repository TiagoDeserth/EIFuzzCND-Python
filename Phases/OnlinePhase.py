import numpy as np
import pandas as pd
import skfuzzy as fuzz
from typing import List

#Importando os outros métodos do projeto
from Models.SupervisedModel import SupervisedModel, _calculate_euclidian_distance
from Models.NotSupervisedModel import NotSupervisedModel
from Structs.Example import Example
from Structs.SPFMiC import SPFMiC

class OnlinePhase:
    #Encapsula a lógica da Fase Online, processando um fluxo de dados

    def __init__(self, supervised_model: SupervisedModel, **kwargs):
        self.supervised_model = supervised_model
        self.not_supervised_model = NotSupervisedModel()
        self.params = kwargs
        self.unk_mem: List[Example] = []
        self.np_label_count = 0

    def run(self, online_data: pd.DataFrame, class_column: str, features: List[str]):
        #Simula a execução da Fase Online, processand o online_data como um fluxo

        print("\n--- Iniciando Fase Online ---")
        results = []

        for index, row, in online_data.iterrows():
            current_time = index + 1
            point = row[features].values
            true_label = row[class_column]
            example = Example(point, true_label, current_time)

            label, success = self.supervised_model.classify_new(example, current_time)

            if not success:
                label = self.not_supervised_model.classify_new(example, self.supervised_model.fuzziness)
            
            example.classified_label = label if label != -1 else 'desconhecido'

            if example.classified_label == 'desconhecido':
                self.unk_mem.append(example)
            
            results.append(example)

            if len(self.unk_mem) >= self.params.get('T', 40):
                print(f"\n[Tempo: {current_time}] Limiar T atingido. Executando Detecção de Novidades...")
                self._multi_class_novelty_detection(current_time)
                self.unk_mem = []

        print("--- Fase Online Concluída ---")
        return results
    
    def _generate_np_label(self):
        #Gera um novo rótulo para um Padrão de Novidade (Novelty Pattern)
        self.np_label_count += 1
        return f"NP-{self.np_label_count}"
    
    def _multi_class_novelty_detection(self, current_time: int):
        #Lógica principal da detecção de novidade
        k_short = self.params.get('k_short', 4)
        if len(self.unk_mem) < k_short:
            return
        
        unk_data = np.array([ex.point for ex in self.unk_mem]).T

        try:
            cntr, u, _, _, _, _, _ = fuzz._cluster.cmeans(
                unk_data, 
                c = k_short,
                m = self.supervised_model.fuzziness,
                error = 0.005,
                maxiter = 1000
            )

        except Exception as e:
            print(f"  -> Erro no Fuzzy C-Means da Detecção de Novidades: {e}")
            return
        
        for i in range(len(cntr)):
            new_centroid = cntr[i]
            points_in_cluster_indices = np.where(np.argmax(u, axis = 0) == i)[0]

            if len(points_in_cluster_indices) < self.params.get('min_weight_online', 5):
                continue

            all_spfmics_known = [spfmic for sublist in self.supervised_model.classifier.values() for spfmic in sublist]
            if not all_spfmics_known:
                continue

            frs = []
            for spfmic_known in all_spfmics_known:
                di = spfmic_known.calculate_radius()
                dj = np.sqrt(np.mean([np.sum((unk_data.T[j] - new_centroid) ** 2) for j in points_in_cluster_indices]))
                dist = _calculate_euclidian_distance(spfmic_known.centroid, new_centroid)
                if dist > 1e-6:
                    frs.append((di + dj) / dist)

            if not frs:
                continue

            min_fr = min(frs)

            if min_fr <= self.params.get('phi', 0.8):
                print(f"  -> Cluster encontrado é extensão de classe conhecida (similaridade: {min_fr:.2f})")
            else:
                new_label_np = self._generate_np_label()
                print(f"  -> NOVIDADE DETECTADA! Cluster gerou o rótulo: {new_label_np} (similaridade: {min_fr:.2f})")

                new_spfmic_np = SPFMiC(
                    centroid = new_centroid,
                    n_examples = len(points_in_cluster_indices),
                    label = new_label_np,
                    alpha = self.supervised_model.alpha,
                    theta = self.supervised_model.theta,
                    time = current_time
                )
                self.not_supervised_model.spfmics.append(new_spfmic_np)