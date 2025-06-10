import pandas as pd
import numpy as np
import skfuzzy as fuzz
from typing import List

#Importando de outros módulos do pacote do código
from Models import SupervisedModel

class OfflinePhase:
    #Encapsula a lógica da fase Offline do algoritmo

    def __init__(self, supervised_model: SupervisedModel):
        self.model = supervised_model

    def run(self, train_data: pd.DataFrame, class_column: str, features: List[str]):
        #Executa a fase Offline completa

        #Args:
            #train_data (pd.DataFrame): O conjunto de dados de treino
            #class_colum (str): O nome da coluna que contém os rótulos das classes
            #features (List[str]): Uma lista com os nomes das colunas de atributos

        print("\nIniciando Fase Offline...")

        grouped_data = train_data.groupby(class_column)
        self.model.known_labels = list(grouped_data.groups.keys())

        print(f"\nClasses conhecidas encontradas: {self.model.known_labels}")
        
        for class_label, group in grouped_data:
            print(f"\nProcessando classe: {class_label} ({len(group)} exemplos)")

            if len(group) < self.model.K:
                print(f"-> Aviso: Número de exemplos ({len(group)}) é menor que K ({self.model.K}). Pulando classe.")
                continue

            data_points = group[features].values.T

            cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
                data_points, c = self.model.K, m = self.model.fuziness, error = 0.005, maxiter = 1000 
            )

            print(f"-> Fuzzy C-Means Completado. {len(cntr)} clusters gerados.")

            spfmics_for_class = self.model._sumarize_clusters_into_spfmics(
                data_points.T, cntr, u, class_label
            )

            self.model.classifier[class_label] = spfmics_for_class
            print(f"-> {len(spfmics_for_class)} SPFMiCs criados para a classe {class_label}.")

        print("\nFase Offline concluída com sucesso.")
        return self.model.classifier