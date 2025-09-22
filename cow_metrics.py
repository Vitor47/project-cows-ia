import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, ConfusionMatrixDisplay

# ======================================
# 1) Carregar dataset
# ======================================
df = pd.read_excel("/home/vitor/projetos-amf/inteligencia-artificial/trabalho_ia/reports/Herd Data 18-08-2024.xlsx")

# Visualização rápida das colunas disponíveis no dataset:
# ['Cow No.', 'Cow Name', 'Group No.', 'Previous Group No.', 'Group Change Date',
#  'Days Since Group Change', 'Sex', 'Days In Pregnancy', 'Insem Date', 'Breed', 
#  'No. of Lact.', 'Days In Milk', 'Cull flagged', 'Birth Date', 'Age Years', 
#  'Days Since Calving', 'Tot. Milk Yest.', 'Milk This Month', ... etc]

# ======================================
# 2) Pré-processamento e Engenharia de Atributos
# ======================================

# Converter data de nascimento para datetime
df["Birth Date"] = pd.to_datetime(df["Birth Date"], errors="coerce")

# Conversão de colunas numéricas
numeric_cols_try = [
    "Tot. Milk Yest.", "Yield This Week ", "Milk Last Month",
    "Milk This Month", "Days In Milk", "Days Since Calving",
    "Calving Interval", "Age Years"
]
for c in numeric_cols_try:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Idade calculada a partir da data de nascimento
if "Birth Date" in df.columns:
    df["Age Years_calc"] = (pd.to_datetime("today") - df["Birth Date"]).dt.days // 365
    df["Age Years"] = df["Age Years"].fillna(df["Age Years_calc"])
    df.drop(columns=["Age Years_calc"], inplace=True)

# Média da produção mensal de leite (se disponível)
monthly_cols = [c for c in df.columns if c.startswith("Monthly Milk")]
if monthly_cols:
    df["Monthly Milk Avg"] = df[monthly_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

# Queda de produção em relação à média mensal
if "Monthly Milk Avg" in df.columns and "Tot. Milk Yest." in df.columns:
    df["Milk Drop"] = df["Monthly Milk Avg"] - df["Tot. Milk Yest."]

# Risco pós-parto baseado nos dias desde o parto
if "Days Since Calving" in df.columns:
    df["Days Since Calving"] = pd.to_numeric(df["Days Since Calving"], errors="coerce")
    df["Postpartum Risk"] = pd.cut(
        df["Days Since Calving"],
        bins=[-1, 30, 90, 365, float("inf")],
        labels=["alto", "medio", "baixo", "muito_baixo"],
        ordered=True
    )

# Categorias de produção de leite (exploração apenas, não usada no modelo)
if "Tot. Milk Yest." in df.columns:
    milk_bins = [0, 10, 20, float("inf")]
    milk_labels = ["baixa", "media", "alta"]
    df["Milk Category"] = pd.cut(df["Tot. Milk Yest."], bins=milk_bins, labels=milk_labels, ordered=True)

# ======================================
# 3) Definição da variável alvo
# ======================================
# Rótulo binário: 1 = risco de baixa produção (<10 litros/dia), 0 = produção normal
if "Tot. Milk Yest." not in df.columns:
    raise ValueError("Coluna 'Tot. Milk Yest.' não encontrada no dataset.")
df["target_low_milk"] = (df["Tot. Milk Yest."] < 10).astype(int)

# ======================================
# 4) Seleção de atributos (evitar vazamento de dados)
# ======================================
leak_cols = {"Tot. Milk Yest.", "Milk Category", "Milk Drop"}  # colunas que não devem ser usadas
candidate_features = [
    "Age Years", "No. of Lact.", "Days In Milk", "Days Since Calving", "Calving Interval",
    "Yield This Week ", "Milk Last Month", "Milk This Month", "Monthly Milk Avg",
    "Pregnant", "Breeding Status", "Breeding State", "Breed", "Sex", "Delivery", "Postpartum Risk"
]
features = [c for c in candidate_features if c in df.columns and c not in leak_cols]

# Base final para ML
df_ml = df[features + ["target_low_milk"]].dropna(subset=["target_low_milk"]).copy()

# Separação entre atributos numéricos e categóricos
numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(df_ml[c])]
categorical_features = [c for c in features if not pd.api.types.is_numeric_dtype(df_ml[c])]

print("Atributos numéricos:", numeric_features)
print("Atributos categóricos:", categorical_features)

# ======================================
# 5) Pipelines de pré-processamento
# ======================================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ======================================
# 6) Divisão treino/teste
# ======================================
X = df_ml.drop(columns=["target_low_milk"])
y = df_ml["target_low_milk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ======================================
# 7) Modelo supervisionado: RandomForest
# ======================================
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ))
])

# Validação cruzada
scores = cross_val_score(rf_model, X, y, cv=5, scoring="roc_auc")
print(f"\n🔎 AUC médio (5-fold CV): {scores.mean():.3f} ± {scores.std():.3f}")

# Treinamento no conjunto de treino
rf_model.fit(X_train, y_train)

# Predições
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

print("\n📊 Relatório de Classificação (RandomForest):")
print(classification_report(y_test, y_pred, target_names=["Produção Normal", "Risco Baixa Produção"]))

# Matriz de Confusão
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Produção Normal", "Risco Baixa Produção"],
    cmap="Blues"
)
plt.title("Matriz de Confusão - RandomForest", fontsize=14, pad=20)
plt.show()

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.title("Curva ROC - RandomForest")
plt.legend()
plt.show()

# ======================================
# 8) Importância das variáveis
# ======================================
rf = rf_model.named_steps["classifier"]
ohe = rf_model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
cat_names = ohe.get_feature_names_out(categorical_features)
all_features = numeric_features + list(cat_names)

feat_imp = pd.Series(rf.feature_importances_, index=all_features).sort_values(ascending=False).head(15)

plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Importância das Variáveis - RandomForest")
plt.xlabel("Importância Relativa")
plt.ylabel("Variáveis")
plt.show()

# ======================================
# 9) Modelo não supervisionado: DBSCAN (Detecção de anomalias)
# ======================================
X_scaled = preprocessor.fit_transform(X)

dbscan = DBSCAN(eps=1.5, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

df_ml["cluster_dbscan"] = clusters
df_ml["cluster_dbscan_label"] = df_ml["cluster_dbscan"].apply(lambda x: "Outlier" if x == -1 else f"Cluster {x}")

# Contagem dos clusters
cluster_counts = df_ml["cluster_dbscan"].value_counts().sort_index()
total = cluster_counts.sum()

print("\n📊 Contagem de clusters DBSCAN:")
cluster_summary = cluster_counts.to_frame(name="Count")
cluster_summary["Percent"] = (cluster_summary["Count"] / total * 100).round(1)
print(cluster_summary)

# Gráfico de distribuição de clusters
plt.figure(figsize=(8,5))
bars = plt.bar(cluster_counts.index, cluster_counts.values, color="skyblue", edgecolor="black")
for bar, idx in zip(bars, cluster_counts.index):
    if idx == -1:
        bar.set_color("salmon")  # Outliers em vermelho
    pct = (cluster_counts[idx] / total) * 100
    plt.text(idx, cluster_counts[idx] + 0.5, f"{cluster_counts[idx]} ({pct:.1f}%)", ha="center")
plt.title("Distribuição dos Clusters DBSCAN (-1 = Outliers)", fontsize=14, pad=15)
plt.xlabel("Cluster")
plt.ylabel("Número de vacas")
plt.show()

# Vacas anômalas
anomalias = df_ml[df_ml["cluster_dbscan"] == -1]
print(f"\n🚨 Vacas anômalas detectadas (cluster -1): {len(anomalias)}")
print(anomalias.head())

# Visualização em 2D com PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_plot = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_plot["Cluster"] = df_ml["cluster_dbscan_label"]

plt.figure(figsize=(8,6))
palette = sns.color_palette("tab10", n_colors=len(cluster_counts))
palette_dict = {f"Cluster {i}": palette[i] for i in range(len(cluster_counts))}
palette_dict["Outlier"] = "red"

sns.scatterplot(
    data=df_plot, x="PC1", y="PC2",
    hue="Cluster", palette=palette_dict,
    alpha=0.7
)
plt.title("Clusters DBSCAN (redução PCA) - Outliers em vermelho", fontsize=14)
plt.show()
