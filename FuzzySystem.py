import pandas as pd

#Íris - Scikit-learn
from sklearn.datasets import load_iris

#Importando as classes dos módulos
from Models.SupervisedModel import SupervisedModel
from Phases.OfflinePhase import OfflinePhase
from Visualization.Plotter import visualize_offline_results

def load_iris_data():
    print("Carregando o dataset Íris...")
    iris = load_iris()

    #Criando o DataFrame
    #iris.data contém os atributos (features)
    #iris.feature_names contém os nomes das colunas de atributos
    df = pd.DataFrame(data = iris.data, columns = iris.feature_names)

    #Adicionando a coluna de classe/rótulo
    #iris.target contém os rótulos numéricos [0, 1, 2]
    df['class'] = iris.target

    return df

if __name__ == '__main__':
    FUZINESS = 2.0

    #Parâmetro específico para o dataset Íris
    #O dataset Íris tem classes bem definidas e pequenas
    #Usar um K grande (como 4 [anteriormente]) para cada classe não é ideal
    #O K vai ser setado para criação de 2 micro-clusters para cada uma das 3 classes
    K_CLUSTERS = 4

    ALPHA = 2.0
    THETA = 1.0
    MIN_WEIGHT_OFFLINE = 0

    #1. Carrefar os dados do Íris
    train_df = load_iris_data()
    print("\nAmostra do Dataset Íris carregado:")
    print(train_df.head())

    print("\nDistribuição das classes:")
    print(train_df['class'].value_counts().sort_index())
    
    #2. Inicializar o modelo com os parâmetros
    supervised_model = SupervisedModel(
        fuziness = FUZINESS,
        k_clusters = K_CLUSTERS,
        alpha = ALPHA,
        theta = THETA,
        min_weight = MIN_WEIGHT_OFFLINE
    )

    #3. Inicializar e executar a fase Offline
    offline_phase = OfflinePhase(supervised_model)

    #Obter os nomes dos atributos (features) dinamicamente do DataFrame
    feature_names = [col for col in train_df.columns if col != 'class']

    model_of_known_classes = offline_phase.run(
        train_data = train_df,
        class_column = 'class',
        features = feature_names
    )

    # #4. Visualizar o resultado do modelo treinado
    # print("\n--- Modelo de Classes Conhecidas (MCC) gerado com o Dataset Íris ---") 
    # for class_label, spfmics in model_of_known_classes.items():

    #     #Opcional: Mapear rótulos numéricos para os nomes das espécies para melhorar a visualização
    #     species_name = load_iris().target_names[class_label]
    #     print(f"Class: {class_label} ({species_name})")
    #     for spfmic in spfmics:
    #         print(f"  -{spfmic}")

    #4. Visualizar os resultados chamando a função nova (importada)
    visualize_offline_results(
        train_data = train_df,
        class_column = 'class',
        features = feature_names,
        model = model_of_known_classes
    )
        