import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer
from sklearn.cluster import DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    calinski_harabasz_score,
    classification_report,
    davies_bouldin_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.dbscan_stability import dbscan_stability

# ======================================
# 1) Carregar dataset
# ======================================
df = pd.read_excel(
    "/home/vitor/projetos-amf/inteligencia-artificial/trabalho_ia/reports/data/Herd Data 18-08-2024.xlsx"
)

# ======================================
# 2) Pré-processamento e engenharia de atributos
# ======================================
df["Birth Date"] = pd.to_datetime(df["Birth Date"], errors="coerce")

numeric_cols_try = [
    "Tot. Milk Yest.",
    "Yield This Week ",
    "Milk Last Month",
    "Milk This Month",
    "Days In Milk",
    "Days Since Calving",
    "Calving Interval",
    "Age Years",
]
for c in numeric_cols_try:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Idade calculada
if "Birth Date" in df.columns:
    df["Age Years_calc"] = (
        pd.to_datetime("today") - df["Birth Date"]
    ).dt.days // 365
    df["Age Years"] = df["Age Years"].fillna(df["Age Years_calc"])
    df.drop(columns=["Age Years_calc"], inplace=True)

# Média mensal
monthly_cols = [c for c in df.columns if c.startswith("Monthly Milk")]
if monthly_cols:
    df["Monthly Milk Avg"] = (
        df[monthly_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    )

# Queda de produção
if "Monthly Milk Avg" in df.columns and "Tot. Milk Yest." in df.columns:
    df["Milk Drop"] = df["Monthly Milk Avg"] - df["Tot. Milk Yest."]

# Risco pós-parto
if "Days Since Calving" in df.columns:
    df["Days Since Calving"] = pd.to_numeric(
        df["Days Since Calving"], errors="coerce"
    )
    df["Postpartum Risk"] = pd.cut(
        df["Days Since Calving"],
        bins=[-1, 30, 90, 365, float("inf")],
        labels=["alto", "medio", "baixo", "muito_baixo"],
        ordered=True,
    )

# Categoria de leite
if "Tot. Milk Yest." in df.columns:
    milk_bins = [0, 10, 20, float("inf")]
    milk_labels = ["baixa", "media", "alta"]
    df["Milk Category"] = pd.cut(
        df["Tot. Milk Yest."], bins=milk_bins, labels=milk_labels, ordered=True
    )

# ======================================
# 3) Variável alvo
# ======================================
df["target_low_milk"] = (df["Tot. Milk Yest."] < 10).astype(int)

# ======================================
# 4) Seleção de atributos
# ======================================
leak_cols = {"Tot. Milk Yest.", "Milk Category", "Milk Drop"}

candidate_features = [
    "Age Years",
    "No. of Lact.",
    "Days In Milk",
    "Days Since Calving",
    "Calving Interval",
    "Yield This Week ",
    "Milk Last Month",
    "Milk This Month",
    "Monthly Milk Avg",
    "Pregnant",
    "Breeding Status",
    "Breeding State",
    "Breed",
    "Sex",
    "Delivery",
    "Postpartum Risk",
]

features = [
    c for c in candidate_features if c in df.columns and c not in leak_cols
]

df_ml = (
    df[features + ["target_low_milk"]]
    .dropna(subset=["target_low_milk"])
    .copy()
)

numeric_features = [
    c for c in features if pd.api.types.is_numeric_dtype(df_ml[c])
]
categorical_features = [
    c for c in features if not pd.api.types.is_numeric_dtype(df_ml[c])
]

print("Atributos numéricos:", numeric_features)
print("Atributos categóricos:", categorical_features)

# ======================================
# 5) Pipelines de pré-processamento
# ======================================
numeric_transformer = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# ======================================
# 6) Divisão treino/teste
# ======================================
X = df_ml.drop(columns=["target_low_milk"])
y = df_ml["target_low_milk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================================
# 7) RandomForest
# ======================================
rf_model = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300, random_state=42, class_weight="balanced"
            ),
        ),
    ]
)

scores = cross_val_score(rf_model, X, y, cv=5, scoring="roc_auc")
print(f"\n🔎 AUC médio (5-fold CV): {scores.mean():.3f} ± {scores.std():.3f}")

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

print("\n📊 Relatório de Classificação (RandomForest):")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Produção Normal", "Risco Baixa Produção"],
    )
)

# Matriz de Confusão
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Produção Normal", "Risco Baixa Produção"],
    cmap="Blues",
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

# Métricas complementares
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("\n📊 Métricas complementares RandomForest:")
print(
    f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1-score: {f1:.3f}"
)

# ======================================
# 8) Importância das variáveis
# ======================================
rf = rf_model.named_steps["classifier"]
ohe = (
    rf_model.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .named_steps["onehot"]
)
cat_names = ohe.get_feature_names_out(categorical_features)
all_features = numeric_features + list(cat_names)

feat_imp = (
    pd.Series(rf.feature_importances_, index=all_features)
    .sort_values(ascending=False)
    .head(15)
)
plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Importância das Variáveis - RandomForest")
plt.xlabel("Importância Relativa")
plt.ylabel("Variáveis")
plt.show()

# ======================================
# 9) SHAP - interpretabilidade
# ======================================
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Features numéricas
numeric_features_transformed = numeric_features

# Features categóricas após one-hot encoding
categorical_features_transformed = (
    preprocessor.named_transformers_["cat"]
    .named_steps["onehot"]
    .get_feature_names_out(categorical_features)
)

# Combinar todas
all_features_transformed = list(numeric_features_transformed) + list(
    categorical_features_transformed
)

explainer_lime = LimeTabularExplainer(
    X_train_transformed,
    feature_names=all_features_transformed,
    class_names=["Normal", "Risco"],
    discretize_continuous=True,
    random_state=42,
)

i = 0  # índice da vaca que você quer explicar
exp = explainer_lime.explain_instance(
    X_test_transformed[i],
    rf_model.named_steps["classifier"].predict_proba,
    num_features=10,  # Top 10 features mais importantes
)

print("\nTop features LIME para a amostra:")
for feat, weight in exp.as_list():
    print(f"{feat}: {weight:.3f}")

# ======================================
# 11) PCA + DBSCAN
# ======================================
df_ml_filtered = df_ml[
    (df_ml["Age Years"] >= 2)
    & (df_ml["Sex"] == "Female")
    & (df_ml["No. of Lact."] > 0)
].copy()
print(f"Total de vacas após filtro: {len(df_ml_filtered)}")

X_filtered = df_ml_filtered.drop(columns=["target_low_milk"])
X_scaled = preprocessor.fit_transform(X_filtered)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

df_plot = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_plot["target_low_milk"] = df_ml_filtered["target_low_milk"].values

# DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
df_plot["Cluster"] = dbscan.fit_predict(X_pca)
df_plot["Cluster_Label"] = df_plot["Cluster"].apply(
    lambda x: "Outlier" if x == -1 else f"Cluster {x}"
)

# Gráfico
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x="PC1",
    y="PC2",
    data=df_plot,
    hue=df_plot["Cluster_Label"],
    style=df_plot["target_low_milk"].map({0: "Normal", 1: "Risco"}),
    palette="tab10",
    markers={"Normal": "o", "Risco": "X"},
    alpha=0.7,
    s=80,
)
plt.title(
    "Clusters DBSCAN (PCA 2D) com Outliers e Risco de Baixa Produção",
    fontsize=14,
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster / Risco", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# Métricas de validação
mask = df_plot["Cluster"] != -1
if mask.sum() > 1 and df_plot["Cluster"].nunique() > 1:
    sil = silhouette_score(X_pca[mask], df_plot.loc[mask, "Cluster"])
    db = davies_bouldin_score(X_pca[mask], df_plot.loc[mask, "Cluster"])
    ch = calinski_harabasz_score(X_pca[mask], df_plot.loc[mask, "Cluster"])
    print("\n📊 Validação dos clusters DBSCAN (sem outliers):")
    print(
        f"Silhouette Score: {sil:.3f}, Davies-Bouldin Index: {db:.3f}, Calinski-Harabasz Index: {ch:.1f}"
    )
else:
    print("\n⚠️ Clusters insuficientes para validação.")

# Outliers
outliers = df_ml_filtered.iloc[df_plot[df_plot["Cluster"] == -1].index]
print(f"\n🚨 Outliers detectados: {len(outliers)}")
print(outliers.head())

# Estabilidade DBSCAN
ari_scores, ari_mean = dbscan_stability(X_pca, eps_values=[0.8, 1.0, 1.2])
print("\n📊 Estabilidade do DBSCAN (ARI comparando eps variado):")
print("Scores individuais:", ari_scores)
print(f"Média ARI: {ari_mean:.3f}")
