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

        #Agrupa as linhas do DataFrame com base no valor da coluna que contém o rótulo
        grouped_data = train_data.groupby(class_column)

        self.model.known_labels = list(grouped_data.groups.keys())

        print(f"\nClasses conhecidas encontradas: {self.model.known_labels}")
        
        #Loop que passará por cada um dos grupos criados 
        for class_label, group in grouped_data:
            print(f"\nProcessando classe: {class_label} ({len(group)} exemplos)")

            if len(group) < self.model.K:
                print(f"-> Aviso: Número de exemplos ({len(group)}) é menor que K ({self.model.K}). Pulando classe.")
                continue

            #Pega os dados do grupo atual, converte para um array NumPy e o transpõe 
            #Isso se dá devido à biblioteca skfuzzy, que espera que os dados de entrada para o Fuzzy C-Means estejam nesse formato específico
            data_points = group[features].values.T
            print(data_points)

            #Aplicação do Fuzzy C-Means
            #O objetivo aqui é encontrar K (2, neste caso) centros de agrupamentos que representam bem os dados da classe atual.
            #A matriz u informa o grau de pertencimento de cada ponto de dado a cada um desses centros
            cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
                data_points,
                c = self.model.K, 
                m = self.model.fuziness, 
                error = 0.005, 
                maxiter = 1000 
            )

            print(f"-> Fuzzy C-Means Completado. {len(cntr)} clusters gerados.")


            #Aqui, é onde se processa os dados brutos oriundos do Fuzzy C-Means, ou do clustering
            #A estrutura retornada pelo FCM não é a desejada, essa chamada aqui do SPFMiC serve para sumarizar esses resultados na estrutura final do modelo (SPFMiCs)
            spfmics_for_class = self.model._sumarize_clusters_into_spfmics(
                data_points.T, 
                cntr, 
                u, 
                class_label
            )

            #Adiciona a lista de SPFMiCs ao dicionário classifier do modelo, usando o rótulo da classe como chave
            self.model.classifier[class_label] = spfmics_for_class

            print(f"-> {len(spfmics_for_class)} SPFMiCs criados para a classe {class_label}.")

        print("\nFase Offline concluída com sucesso.")

        #Retorna o dicionário classifier completo e preenchido. Entrega o modelo treinado de volta ao arquivo inicial
        return self.model.classifier