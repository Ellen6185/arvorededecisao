import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

# Definir as colunas
columns = [
    "Class", "Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium",
    "Total Phenols", "Flavanoids", "Nonflavanoid Phenols", "Proanthocyanins",
    "Color Intensity", "Hue", "OD280/OD315 of Diluted Wines", "Proline"
]

# Carregar os dados
df = pd.read_csv("C:\\Sétimo Período\\Inteligêpipncia Artificial\\arvore de decisao\\bancodedados\\wine.data", names=columns)

# Tratamento dos dados - Substituindo valores ausentes com a média das colunas numéricas
df.fillna(df.mean(), inplace=True)

# Dividir os dados em variáveis independentes (X) e dependentes (y)
X = df.drop(columns=["Class"])
y_class = df["Class"]
y_regress = df[["Alcohol", "Color Intensity"]]  # Saída múltipla para regressão

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir os dados em treino e teste
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_regress, test_size=0.2, random_state=42)

# Criar pesos amostrais para a classificação (balanceamento)
sample_weights_class = compute_sample_weight(class_weight="balanced", y=y_train_c)

# Otimização de Hiperparâmetros (GridSearchCV)
param_grid_tree = {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]}

print("Treinando modelo de classificação...")

# ----------------------- Classificação ------------------------
# Sem Balanceamento
grid_tree_c = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_tree, cv=5, scoring="accuracy")
grid_tree_c.fit(X_train_c, y_train_c)
best_tree_c = grid_tree_c.best_estimator_

# Com Balanceamento
grid_tree_c_bal = GridSearchCV(DecisionTreeClassifier(class_weight="balanced", random_state=42), param_grid_tree, cv=5, scoring="accuracy")
grid_tree_c_bal.fit(X_train_c, y_train_c)
best_tree_c_bal = grid_tree_c_bal.best_estimator_

print("Treinando modelo de regressão...")

# ----------------------- Regressão ------------------------
# Sem Balanceamento
grid_tree_r = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid_tree, cv=5, scoring="r2")
grid_tree_r.fit(X_train_r, y_train_r)
best_tree_r = grid_tree_r.best_estimator_

# Com Balanceamento (Usando sample_weight)
grid_tree_r_bal = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid_tree, cv=5, scoring="r2")
grid_tree_r_bal.fit(X_train_r, y_train_r)
best_tree_r_bal = grid_tree_r_bal.best_estimator_

# ----------------------- Poda de Árvore de Decisão ------------------------
# Obter caminho de complexidade de poda
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train_c, y_train_c)
path = clf.cost_complexity_pruning_path(X_train_c, y_train_c)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Plotando Impurezas vs Alfa Efetivo
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("Alfa efetivo")
ax.set_ylabel("Impureza total das folhas")
ax.set_title("Impureza total vs Alfa efetivo para o conjunto de treinamento")
plt.show()

# Treinar modelos podados para diferentes valores de ccp_alpha
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train_c, y_train_c)
    clfs.append(clf)

# Avaliar os modelos treinados (Acurácia)
train_scores = [clf.score(X_train_c, y_train_c) for clf in clfs]
test_scores = [clf.score(X_test_c, y_test_c) for clf in clfs]

# Plotando Acurácia vs Alfa Efetivo
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], train_scores[:-1], marker="o", label="Acurácia de treinamento")
ax.plot(ccp_alphas[:-1], test_scores[:-1], marker="o", label="Acurácia de teste")
ax.set_xlabel("Alfa efetivo")
ax.set_ylabel("Acurácia")
ax.set_title("Acurácia vs Alfa efetivo para treinamento e teste")
ax.legend()
plt.show()

# Avaliar o modelo com melhor alfa (maximizar a acurácia)
best_alpha = ccp_alphas[np.argmax(test_scores)]
print(f"Melhor valor de ccp_alpha: {best_alpha}")

# Treinar árvore com o melhor valor de alfa
best_tree_pruned = DecisionTreeClassifier(random_state=0, ccp_alpha=best_alpha)
best_tree_pruned.fit(X_train_c, y_train_c)

# ----------------------- Avaliação ------------------------
# ----------------------- Classificação ------------------------
print("Avaliando modelos de classificação...")

# Predições
y_pred_tree_c = best_tree_c.predict(X_test_c)
y_pred_tree_c_bal = best_tree_c_bal.predict(X_test_c)
y_pred_tree_pruned = best_tree_pruned.predict(X_test_c)

# Métricas
acc_tree_c = accuracy_score(y_test_c, y_pred_tree_c)
acc_tree_c_bal = accuracy_score(y_test_c, y_pred_tree_c_bal)
acc_tree_pruned = accuracy_score(y_test_c, y_pred_tree_pruned)

print(f"\nAcuracia Arvore Classificacao (Sem Balanceamento): {acc_tree_c:.2f}")
print(f"Acuracia Arvore Classificacao (Com Balanceamento): {acc_tree_c_bal:.2f}")
print(f"Acuracia Arvore Classificacao (Com Poda): {acc_tree_pruned:.2f}")

print("\nRelatorio de Classificacao - Sem Balanceamento")
print(classification_report(y_test_c, y_pred_tree_c))

print("\nRelatorio de Classificacao - Com Balanceamento")
print(classification_report(y_test_c, y_pred_tree_c_bal))

print("\nRelatorio de Classificacao - Com Poda")
print(classification_report(y_test_c, y_pred_tree_pruned))

# Matriz de Confusão
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.heatmap(confusion_matrix(y_test_c, y_pred_tree_c), annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Matriz de Confusão - Sem Balanceamento")

sns.heatmap(confusion_matrix(y_test_c, y_pred_tree_c_bal), annot=True, fmt="d", cmap="Reds", ax=axes[1])
axes[1].set_title("Matriz de Confusão - Com Balanceamento")

sns.heatmap(confusion_matrix(y_test_c, y_pred_tree_pruned), annot=True, fmt="d", cmap="Greens", ax=axes[2])
axes[2].set_title("Matriz de Confusão - Com Poda")

plt.show()

# ----------------------- Regressão ------------------------
print("Avaliando modelos de regressão...")

# Predições
y_pred_tree_r = best_tree_r.predict(X_test_r)
y_pred_tree_r_bal = best_tree_r_bal.predict(X_test_r)

# Métricas
mae_tree_r = mean_absolute_error(y_test_r, y_pred_tree_r)
mae_tree_r_bal = mean_absolute_error(y_test_r, y_pred_tree_r_bal)
r2_tree_r = r2_score(y_test_r, y_pred_tree_r, multioutput='uniform_average')
r2_tree_r_bal = r2_score(y_test_r, y_pred_tree_r_bal, multioutput='uniform_average')

print(f"\nErro Medio Absoluto (MAE) - Sem Balanceamento: {mae_tree_r:.2f}")
print(f"Erro Medio Absoluto (MAE) - Com Balanceamento: {mae_tree_r_bal:.2f}")
print(f"Coeficiente de Determinacao (R²) - Sem Balanceamento: {r2_tree_r:.2f}")
print(f"Coeficiente de Determinacao (R²) - Com Balanceamento: {r2_tree_r_bal:.2f}")

# Comparação dos erros da Regressão
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

sns.histplot(y_test_r - y_pred_tree_r, bins=20, kde=True, color="blue", ax=ax[0])
ax[0].set_title("Distribuição dos Erros - Sem Balanceamento")

sns.histplot(y_test_r - y_pred_tree_r_bal, bins=20, kde=True, color="red", ax=ax[1])
ax[1].set_title("Distribuição dos Erros - Com Balanceamento")

plt.show()

# ----------------------- Plotagem das Árvores ------------------------
print("Plotando árvores de decisão...")

plt.figure(figsize=(15, 10))
plot_tree(best_tree_c, feature_names=df.drop(columns=["Class"]).columns, class_names=["1", "2", "3"], filled=True)
plt.title("Árvore de Decisão - Classificação (Sem Balanceamento)")
plt.show()

plt.figure(figsize=(15, 10))
plot_tree(best_tree_c_bal, feature_names=df.drop(columns=["Class"]).columns, class_names=["1", "2", "3"], filled=True)
plt.title("Árvore de Decisão - Classificação (Com Balanceamento)")
plt.show()

plt.figure(figsize=(15, 10))
plot_tree(best_tree_pruned, feature_names=df.drop(columns=["Class"]).columns, class_names=["1", "2", "3"], filled=True)
plt.title("Árvore de Decisão Podada (Com ccp_alpha)")
plt.show()



