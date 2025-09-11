import pandas as pd
import os
import math
from collections import Counter

# *Importando as classes dos módulos
from Models.SupervisedModel import SupervisedModel
from Phases.OfflinePhase import OfflinePhase
from Phases.OnlinePhase import OnlinePhase
from Visualization.Plotter import visualize_offline_results, plot_performance_over_time
from Structs.Example import Example
from Metrics import Metrics

# def separate_data_offline_online(filepath: str, num_classes_offline: int, num_instances_offline: int, class_column: str = 'classes'):
#     # *Faz a separação do conjunto de dados original em Offline e Online, como descrito na dissertação do Lucas (Tabela 5.1)
#     print(f"--- Iniciando a separação do dataset: {filepath} ---")

#     df_complete = pd.read_csv(filepath)
#     print(f"Total de instâncias carregadas: {len(df_complete)}")

#     classes_uniques = sorted(df_complete[class_column].unique())
#     classes_offline_labels = classes_uniques[:num_classes_offline]
#     print(f"Classes selecionadas para a fase Offline: {classes_offline_labels}")

#     # *Cria o DataFrame para ser usado na fase Offline
#     df_instances_offline = df_complete[df_complete[class_column].isin(classes_offline_labels)]
#     df_offline = df_instances_offline.head(num_instances_offline).copy()

#     # *Cria o Dataframe para ser usado na fase Online
#     indices_used_offline = df_offline.index
#     df_online = df_complete.drop(indices_used_offline).sample(frac = 1, random_state = 42).reset_index(drop = True)

#     print(f"\nSeparação feita:")
#     print(f"-> Tamanho do conjunto Offline: {len(df_offline)} instâncias")
#     print(f"-> Tamanho do conjunto Online: {len(df_online)} instâncias")

#     return df_offline, df_online

def separate_data_offline_online(filepath: str, num_instances_offline: int):
    """
    Separa o conjunto de dados em duas partes: Offline e Online.
    A separação é feita de forma simples e sequencial:
        - As primeiras 'num_instances_offline' linhas são usadas para a fase Offline.
        - Todo o restante do ficheiro é usado para a fase Online, mantendo a ordem original.
    """

    print(f"--- Iniciando a separação sequencial do dataset: {filepath} ---")

    #1. Carrega o dataset completo
    df_complete = pd.read_csv(filepath)
    print(f"Total de instâncias carregadas: {len(df_complete)}")

    # df_complete.dropna(inplace=True)

    #2. Seleciona as primeiras 'num_instances_offline' para o treino
    #   O método .loc[] é usado para selecionar linhas por sua posição inteira
    df_offline = df_complete.iloc[:num_instances_offline].copy()

    #3. Seleciona todo o restante para a fase Online
    #   Começa da linha 'num_instances_offline' e vai até o final do dataset
    df_online = df_complete.iloc[num_instances_offline:].copy()

    #Reajusta os índices do dataset Online para começar do 0, o que é uma boa prática
    df_online.reset_index(drop = True, inplace = True)

    print(f"\nSeparação feita:")
    print(f"-> Tamanho do conjunto Offline: {len(df_offline)} instâncias (Linhas 0 a {num_instances_offline - 1})")
    print(f"-> Tamnho do conjunto Online: {len(df_online)} instâncias (Linhas {num_instances_offline} em diante)")

    return df_offline, df_online

if __name__ == '__main__':
    # *Parâmetros paar o RBF
    DATASET_FILE = 'Data/RBF3_40000.csv' # *Altere aqui o nome do dataset
    DATASET_NAME = 'RBF'
    CLASS_COLUMN = 'class'

    # *Parâmetros do modelo
    FUZZINESS = 2.0
    K_CLUSTERS = 4
    ALPHA = 2.0
    THETA = 1.0
    MIN_WEIGHT_OFFLINE = 15

    # *Parâmetros para a separação do dataset 
    # CLASSES_OFFLINE_RBF = 3
    INSTANCES_OFFLINE_RBF = 2000

    online_params = {
        'T': 40,
        'k_short': 4,
        'phi': 0.5,
        'min_weight_online': 15
    }

    # rbf_offline_df, rbf_online_df = separate_data_offline_online(
    #     filepath = DATASET_FILE,
    #     num_classes_offline = CLASSES_OFFLINE_RBF,
    #     num_instances_offline = INSTANCES_OFFLINE_RBF,
    #     class_column = CLASS_COLUMN
    # )

    # *1. Separar o dataset em Offline e Online
    rbf_offline_df, rbf_online_df = separate_data_offline_online(
        filepath = DATASET_FILE,
        num_instances_offline = INSTANCES_OFFLINE_RBF
    )

    # *2. Inicializar o modelo com os parâmetros
    supervised_model = SupervisedModel(
        fuzziness = FUZZINESS,
        k_clusters = K_CLUSTERS,
        alpha = ALPHA,
        theta = THETA,
        min_weight = MIN_WEIGHT_OFFLINE
    )

    # *3. Inicializar e executar a Fase Offline com o conjunto de treino correto
    offline_phase = OfflinePhase(supervised_model)
    feature_names = [col for col in rbf_offline_df.columns if col != CLASS_COLUMN]

    model_of_kown_classes = offline_phase.run(
        train_data = rbf_offline_df,
        class_column = CLASS_COLUMN,
        features = feature_names
    )

    # *4. Exibir o resultado do modelo treinado no console
    print("\n--- Modelo de Classes Conhecidas (MCC) gerado ---")
    for class_label, spfmics in model_of_kown_classes.items():
        print(f"Classe: {class_label}")
        for spfmic in spfmics:
            print(f" - {spfmic}")

    # *5. Visualizar os resultados graficamente
    print("\nIniciando visualização gráfica...")
    visualize_offline_results(
        train_data = rbf_offline_df,
        class_column = CLASS_COLUMN,
        features = feature_names,
        model = model_of_kown_classes,
        dataset_name = DATASET_NAME
    )

    # *RBF pronto para o uso na fase Online, uma prévia do mesmo:
    print(f"\nO conjunto de dados online, com {len(rbf_online_df)} instâncias")

    # *6. Fase Online
    print("\nFeche o gráfico para iniciar a Fase Online...")
    
    online_phase = OnlinePhase(supervised_model, **online_params)

    # online_results = online_phase.run(
    #     rbf_online_df,
    #     CLASS_COLUMN, 
    #     feature_names
    # )

    online_results = []
    performance_timestamps = []
    performance_accuracies = []
    evaluation_interval = 1000

    for index, row in rbf_online_df.iterrows():
        current_time = index + 1
        point = row[feature_names].values
        true_label = row[CLASS_COLUMN]
        example = Example(point, true_label, current_time)

        processed_example = online_phase.process_example(example)
        online_results.append(processed_example)

        if current_time % evaluation_interval == 0:
            print(f"[Tempo: {current_time}] Medindo desempenho...")

            true_so_far = [ex.true_label for ex in online_results]
            classified_so_far = [ex.classified_label for ex in online_results]

            #eval_true = [t for t, c in zip(true_so_far, classified_so_far) if c != 'desconhecido']
            #eval_classified = [c for c in classified_so_far if c != 'desconhecido']

            #eval_true = [str(t) for t, c in zip(true_so_far, classified_so_far) if c != 'desconhecido' and not (isinstance(t, float) and math.isnan(t))]
            #eval_classified = [str(c) for c in classified_so_far if c != 'desconhecido' and not (isinstance(c, str) and c.lower() == 'nan')]

            evaluation_pairs = []
            for t, c in zip(true_so_far, classified_so_far):
                is_true_label_valid = t is not None and not (isinstance(t, float) and math.isnan(t))

                if c != 'desconhecido' and is_true_label_valid:
                    evaluation_pairs.append((str(t), str(c)))

            if evaluation_pairs:
                eval_true, eval_classified = zip(*evaluation_pairs)
                eval_true, eval_classified = list(eval_true), list(eval_classified)
            else:
                eval_true, eval_classified = [], []

            if eval_true:
                all_labels_so_far = list(set(str(l) for l in true_so_far if l is not None and not math.isnan(l)))
                metrics_calculator = Metrics(all_labels_so_far) # TODO: ver depois
                metrics_calculator.calculate_metrics(eval_true, eval_classified)

                performance_timestamps.append(current_time)
                performance_accuracies.append(metrics_calculator.accuracy)

    # *7. Análise dos resultados online
    print("\n--- Análise dos Resultados da Fase Online ---")

    true_labels = [ex.true_label for ex in online_results]
    classified_labels = [ex.classified_label for ex in online_results]

    evaluation_pairs = []
    for t, c in zip(true_labels, classified_labels):
        is_true_label_valid = t is not None and not (isinstance(t, float) and math.isnan(t))
        if c != 'desconhecido' and is_true_label_valid:
            evaluation_pairs.append((str(t), str(c)))

    if evaluation_pairs:
        eval_true_labels, eval_classified_labels = zip(*evaluation_pairs)
        eval_true_labels, eval_classified_labels = list(eval_true_labels), list(eval_classified_labels)
    else:
        eval_true_labels, eval_classified_labels = [], []
    
    #cleaned_true = [str(t) for t in true_labels if t is not None and not (isinstance(t, float) and math.isnan(t))]
    #cleaned_classified = [str(c) for c in classified_labels if c is not None and not (isinstance(c, float) and math.isnan(c))]

    # Agora, filtramos os 'desconhecidos' a partir das listas já limpas
    #eval_true_labels = [t for t, c in zip(cleaned_true, cleaned_classified) if c != 'desconhecido']
    #eval_classified_labels = [c for c in cleaned_classified if c != 'desconhecido']

    #all_possible_labels = list(set(str(l) for l in true_labels if l is not None and not math.isnan(l)) + list(set(classified_labels)))

    true_set = set(str(l) for l in true_labels if l is not None and not (isinstance(l, float) and math.isnan(l)))
    classified_set = set(str(c) for c in classified_labels if c is not None)
    all_possible_labels = list(true_set.union(classified_set))

    metrics_calculator = Metrics(all_possible_labels)
    if eval_true_labels:
        metrics_calculator.calculate_metrics(eval_true_labels, eval_classified_labels)
        metrics_calculator.print_report(eval_true_labels)

    print(f"\nTotal de exemplos no fluxo online: {len(online_results)}")
    print(f"Total de exemplos classificados: {len(eval_true_labels)}")
    print(f"Total de exemplos não classificados ('desconhecido'): {classified_labels.count('desconhecido')}")
    
    print("\nContagem geral de classificações:")
    print(Counter(classified_labels))

    print("\nModelo de Novidades (MCD) final:")
    
    if not online_phase.not_supervised_model.spfmics:
        print("Nenhum cluster de novidade foi criado")
    else:
        for spfmic in online_phase.not_supervised_model.spfmics:
            print(f" - {spfmic}")

    # 8. Visualizar o desempenho ao longo do tempo
    if performance_timestamps:
        plot_performance_over_time(performance_timestamps, performance_accuracies, DATASET_NAME)


    # # *Classificados conhecidos
    # popular_classifieds = [(rt, rc) for rt, rc in zip(true_labels, classified_labels) if isinstance(rc, (int, float)) and rc != -1]

    # known_success = sum(1 for rt, rc in popular_classifieds if rt == rc)

    # print(f"Total de exemplos no fluxo online: {len(online_results)}")
    
    # if popular_classifieds:
    #     print(f"Acurácia nas instâncias classificados (não desconhecidas/NP): {known_success / len(popular_classifieds) * 100:.2f}%")
    # print("\nContagem de clasificações:")
    # print(Counter(classified_labels))
    # print("\nModelo de Novidades (MCD) final:")
    # if not online_phase.not_supervised_model.spfmics: 
    #     print("Nenhum cluster de novidade foi criado")
    # else:
    #     for spfmic in online_phase.not_supervised_model.spfmics:
    #         print(f" - {spfmic}")




        