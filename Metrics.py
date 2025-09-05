import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score

class Metrics:
    def __init__(self, all_labels):
        """
        Inicializa o objeto de métricas.
        'all_labels' deve ser uma lista de todos os rótulos possíveis (conhecidos e novidades).
        """
        # self.labels = sorted(list(set(all_labels)))
        self.labels = sorted(list(set(str(label) for label in all_labels if label is not None)))
        self.cm = None
        self.accuracy = 0.0
        self.class_accuracy = {}

    def calculate_metrics(self, true_labels, classified_labels):
        """
        Calcula a matriz de confusão, a acurácia geral e a acurácia por classe.
        """
        # Converte os rótulos para string aqui também para garantir consistência
        true_labels_str = [str(label) for label in true_labels]
        classified_labels_str = [str(label) for label in classified_labels]

        current_labels = sorted(list(set(true_labels_str + classified_labels_str)))
        complete_label_set = sorted(list(set(self.labels + current_labels)))

        self.cm = confusion_matrix(true_labels, classified_labels, labels=complete_label_set)
        self.accuracy = accuracy_score(true_labels, classified_labels)

        # Calcula a acurácia para cada classe
        for i, label in enumerate(complete_label_set):
            true_positives = self.cm[i, i]
            total_for_class = np.sum(self.cm[i, :])
            if total_for_class > 0:
                self.class_accuracy[label] = true_positives / total_for_class
            else:
                # Se uma classe nunca apareceu nos rótulos verdadeiros, a sua acurácia é 0.
                if label in set(true_labels):
                    self.class_accuracy[label] = 0.0

    def print_report(self, true_labels):
        """
        Imprime um relatório de desempenho detalhado, semelhante ao da versão Java.
        """
        print("\n--- Relatório de Desempenho ---")
        print(f"Acurácia Geral: {self.accuracy * 100:.2f}%")

        print("\nAcurácia por Classe:")
        # Itera sobre os rótulos ordenados para uma apresentação consistente
        for label in sorted(self.class_accuracy.keys()):
             # Apenas mostra classes que estavam nos dados de teste
            if label in set(true_labels):
                 print(f"  - Classe {label}: {self.class_accuracy[label] * 100:.2f}%")

        print("\nMatriz de Confusão:")
        
        # Extrai os rótulos que realmente aparecem na matriz para a impressão
        labels_in_cm = [l for l in self.cm[0] if l in set(true_labels) or l in set(self.cm[1])]
        
        # Imprime o cabeçalho
        header = "      " + " ".join([f"{str(label):>5}" for label in labels_in_cm])
        print(header)
        print("     " + "-" * (len(header) - 5))
        
        # Imprime as linhas
        for i, row in enumerate(self.cm):
            # Apenas imprime linhas para classes que existem nos rótulos verdadeiros
            if self.cm[i][0] in set(true_labels):
                row_str = f"{str(self.cm[i][0]):>5} |" + " ".join([f"{val:>5}" for val in row])
                print(row_str)