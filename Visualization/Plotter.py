import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import Dict, List

from Structs.SPFMiC import SPFMiC

def visualize_offline_results(train_data: pd.DataFrame, class_column: str, features: List[str],
                              model: Dict[int, List[SPFMiC]], dataset_name: str):
    
    #Visualiza os resultados da fase Offline usando PCA para redução de dimensionalidade
    #Args:
    #     train_data (pd.DataFrame): O DataFrame original usado no treino
    #     class_column (str): Nome da coluna de classe
    #     features (List[str]): Nomes das colunas de atributos
    #     model (Dict): O modelo treinado (MCC) contendo os SPFMiCs

    print("\nGerando visualização dos resultados a partir de 'Visualization/Plotter.py...")

    #1. Aplicar o PCA para reduzir os dados de 4D para 2D
    pca = PCA(n_components = 2)
    data_2d = pca.fit_transform(train_data[features])

    plt.figure(figsize = (12, 8))

    #2. Plotar os pontos de dados originais
    # target_names = load_iris().target_names
    # colors = ['navy', 'turquoise', 'darkorange']

    unique_labels = sorted(train_data[class_column].unique())

    colors = plt.cm.get_cmap('viridis', len(unique_labels))

    # for i, target_name in enumerate(target_names):
    #     plt.scatter(
    #         data_2d[train_data[class_column] == i, 0],
    #         data_2d[train_data[class_column] == i, 1],
    #         color = colors[i],
    #         alpha = 0.8,
    #         lw = 2,
    #         label = target_name
    #     )

    for i, label in enumerate(unique_labels):
        plt.scatter(
            data_2d[train_data[class_column] == label, 0],
            data_2d[train_data[class_column] == label, 1],
            color = colors(i),
            alpha = 0.8,
            label = f'Classe {label}'
        )

    #3. Extrair, transformar e plotar os centroide dos SPFMiCs
    all_centroids = []
    for class_label, spfmics in model.items():
        for spfmic in spfmics:
            all_centroids.append(spfmic.centroid)
    
    if all_centroids:
        #Aplicar a MESMA transformação PCA aos centroides
        centroids_2d = pca.transform(np.array(all_centroids))

        plt.scatter(
            centroids_2d[:, 0],
            centroids_2d[:, 1],
            marker = 'x',
            s = 200,
            linewidths = 3,
            color = 'red',
            label = 'Centroides (SPFMiCS)'
        )

    plt.legend(loc = 'best', shadow = False, scatterpoints = 1)
    plt.title('Resultado da Fase Offline do EIFuzzCND no Dataset {dataset_name} (com PCA)')
    plt.xlabel('Componente principal 1')
    plt.ylabel('Componente principal 2')
    plt.grid(True)
    plt.show()

def plot_performance_over_time(timestamps, accuracies, dataset_name):
    plt.figure(figsize = (12, 6))
    plt.plot(timestamps, accuracies, marker = 'o', linestyle = '-', label = 'Acurácia do EIFuzzCND')

    plt.title(f"Desempenho ao longo do tempo do EIFuzzCND no Dataset {dataset_name}")
    plt.xlabel("Número de Instâncias Processadas")
    plt.ylabel("Acurácia")
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 1.05)

    print("\nExibindo gráfico de desempenho ao londo do tempo...")
    plt.show()