import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar los datos
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

# Dividir los datos en entrenamiento, validación y prueba
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)  # 80% entrenamiento, 20% validación

def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Evitar log(0)

def information_gain(y, X_column):
    parent_entropy = entropy(y)
    unique_values, counts = np.unique(X_column, return_counts=True)
    weighted_entropy = np.sum([
        (counts[i] / counts.sum()) * entropy(y[X_column == v])
        for i, v in enumerate(unique_values)
    ])
    return parent_entropy - weighted_entropy

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature          # Índice de la característica
        self.threshold = threshold      # Valor de umbral para dividir
        self.left = left                # Subárbol izquierdo
        self.right = right              # Subárbol derecho
        self.value = value              # Valor si es hoja

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # Condiciones de parada
        if (depth >= self.max_depth 
            or num_labels == 1 
            or num_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Encontrar la mejor división
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Dividir el conjunto de datos
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feature in range(self.n_features_):
            X_column = X[:, feature]
            unique_values = np.unique(X_column)
            for threshold in unique_values:
                # División binaria
                left = X_column <= threshold
                right = X_column > threshold
                if len(y[left]) == 0 or len(y[right]) == 0:
                    continue
                gain = information_gain(y, X_column <= threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_threshold = threshold

        return split_idx, split_threshold

    def _most_common_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def prune(self, X_val, y_val):
        """
        Implementa la poda de error reducido utilizando el conjunto de validación.
        """
        self._prune_tree(self.root, X_val, y_val)

    def _prune_tree(self, node, X_val, y_val):
        if node.is_leaf_node():
            return

        # Dividir el conjunto de validación según el umbral actual
        if node.feature is not None:
            left_indices = X_val[:, node.feature] <= node.threshold
            right_indices = X_val[:, node.feature] > node.threshold

            if node.left:
                self._prune_tree(node.left, X_val[left_indices], y_val[left_indices])
            if node.right:
                self._prune_tree(node.right, X_val[right_indices], y_val[right_indices])

        # Intentar podar este nodo
        # Guardar referencias a los subárboles actuales
        left_subtree = node.left
        right_subtree = node.right
        current_value = node.value

        # Convertir este nodo en una hoja
        node.left = None
        node.right = None
        node.value = self._most_common_label(y_val) if len(y_val) > 0 else current_value

        # Evaluar el desempeño después de la poda
        pruned_accuracy = self.current_accuracy(X_val, y_val)

        # Calcular la precisión sin podar (antes de podar)
        # Restaurar el nodo si la poda empeora la precisión
        if len(y_val) > 0:
            # Restaurar el nodo para calcular la precisión sin podar
            node.left = left_subtree
            node.right = right_subtree
            node.value = current_value
            unpruned_accuracy = self.current_accuracy(X_val, y_val)

            if pruned_accuracy >= unpruned_accuracy:
                # Mantener la poda
                node.left = None
                node.right = None
                node.value = self._most_common_label(y_val)
            else:
                # Restaurar el nodo
                node.left = left_subtree
                node.right = right_subtree
                node.value = current_value

    def current_accuracy(self, X_val, y_val):
        y_pred = self.predict(X_val)
        return accuracy_score(y_val, y_pred)

# Convertir DataFrame a NumPy arrays
X_train_np = X_train.values
y_train_np = y_train
X_val_np = X_val.values
y_val_np = y_val
X_test_np = X_test.values
y_test_np = y_test

# Inicializar y entrenar el árbol
tree = DecisionTree(max_depth=10)
tree.fit(X_train_np, y_train_np)

# Evaluar antes de la poda
print("=== Antes de la poda ===")
y_pred = tree.predict(X_test_np)
accuracy = accuracy_score(y_test_np, y_pred)
print(f"Precisión del modelo: {accuracy:.2f}")
print("\nReporte de clasificación:")
print(classification_report(y_test_np, y_pred))
print("Matriz de confusión:")
print(confusion_matrix(y_test_np, y_pred))

# Poda el árbol utilizando el conjunto de validación
tree.prune(X_val_np, y_val_np)

# Evaluar después de la poda
print("\n=== Después de la poda ===")
y_pred_pruned = tree.predict(X_test_np)
accuracy_pruned = accuracy_score(y_test_np, y_pred_pruned)
print(f"Precisión del modelo después de la poda: {accuracy_pruned:.2f}")
print("\nReporte de clasificación después de la poda:")
print(classification_report(y_test_np, y_pred_pruned))
print("Matriz de confusión después de la poda:")
print(confusion_matrix(y_test_np, y_pred_pruned))