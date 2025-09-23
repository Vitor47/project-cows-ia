# 🐄 Sistema de Rastreamento e Monitoramento de Vacas Leiteiras com Análise Preditiva

Projeto desenvolvido na disciplina **Inteligência Artificial II – 2025/02**  
Faculdade Antonio Meneghetti (AMF)  
Autor: **Vitor Mateus Miolo**  

## 📌 Descrição
Este projeto utiliza **técnicas de machine learning** para prever riscos na produção de leite, saúde e descarte de vacas leiteiras.  
Além disso, aplica **clusterização (DBSCAN)** para identificar grupos ocultos e **modelos de regressão** para previsão de produtividade.

## 📂 Estrutura do Projeto
```sh
├── reports/
    ├── data/
        ├── Herd Data 18-08-2024.xlsx # Conjunto de dados (Dairy Farm Dataset - Kaggle 2024)
        ├── Relatório Técnico Cow Metrics AI.docx # Relatório Técnico gerado em word
    ├── figures/         # Gráficos gerados (Matriz Confusão, ROC, etc.)
├── requirements.txt     # Dependências do projeto
├── cow_metrics.py  # Script de execução
└── README.md            # Este arquivo
```

## ⚙️ Instalação

### 1. Clone o repositório
```bash
git clone https://github.com/Vitor47/project-cows-ia.git
cd project-cows-ia
````

### 2. Crie e ative um ambiente virtual

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Instale as dependências de desenvolvimento

```bash
pip install -r requirements-dev.txt
```

## ▶️ Execução

### Rodar o pipeline completo

```bash
python cow_metrics.py
```

## 📊 Resultados

* **Classificação:** RandomForest obteve AUC = 1.00
* **Clusterização:** DBSCAN identificou grupos distintos de vacas e outliers
* **Variáveis importantes:** produção semanal/mensal de leite, dias em lactação e idade

Gráficos principais:

* ✅ Matriz de confusão
* ✅ Curva ROC
* ✅ Importância das variáveis
* ✅ Clusters DBSCAN

## 📌 Próximos Passos

* Integração com sensores IoT
* Validação em múltiplas fazendas
* Desenvolvimento de dashboard interativo

## 📜 Licença

Este projeto utiliza dados do **Dairy Farm Dataset (Kaggle, 2024)** sob licença Creative Commons Attribution.

## 🙌 Créditos

* **Bibliotecas:** scikit-learn, pandas, numpy, seaborn, matplotlib
* **Orientação:** Prof. da disciplina Inteligência Artificial II – AMF
* **Apoio na estruturação:** ChatGPT (OpenAI, 2025)
