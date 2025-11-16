import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Configurações de estilo
plt.style.use('default')
sns.set_palette("husl")

# Carregar os dados
# Supondo que o arquivo esteja no mesmo diretório
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Visualizar as primeiras linhas
print("Primeiras 5 linhas do dataset:")
print(df.head())
print("\n" + "="*50)

# Informações sobre o dataset
print("\nInformações do dataset:")
print(df.info())
print("\n" + "="*50)

# Estatísticas descritivas
print("\nEstatísticas descritivas:")
print(df.describe())
print("\n" + "="*50)

# Verificar valores nulos
print("\nValores nulos por coluna:")
print(df.isnull().sum())
print("\n" + "="*50)

# Separar features para clustering
# Vamos usar todas as features numéricas exceto a target (DEATH_EVENT)
features_for_clustering = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
                          'ejection_fraction', 'high_blood_pressure', 'platelets',
                          'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']

X = df[features_for_clustering]

print("Features selecionadas para clustering:")
print(X.columns.tolist())
print(f"\nShape dos dados: {X.shape}")
print("\n" + "="*50)

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Dados normalizados - primeiras 5 linhas:")
print(X_scaled[:5])
print("\n" + "="*50)

def find_optimal_clusters(X, max_k=10):
    """
    Encontra o número ótimo de clusters usando o método do cotovelo e silhouette score
    """
    wcss = []  # Within-Cluster Sum of Square
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        
        # Calcula silhouette score
        if k > 1:  # silhouette score requer pelo menos 2 clusters
            score = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(score)
    
    return k_range, wcss, silhouette_scores

# Encontrar número ótimo de clusters
k_range, wcss, silhouette_scores = find_optimal_clusters(X_scaled)

# Plotar os resultados
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Método do cotovelo
ax1.plot(k_range, wcss, 'bo-')
ax1.set_xlabel('Número de Clusters (k)')
ax1.set_ylabel('WCSS')
ax1.set_title('Método do Cotovelo')
ax1.grid(True, alpha=0.3)

# Silhouette Score
ax2.plot(k_range, silhouette_scores, 'ro-')
ax2.set_xlabel('Número de Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Determinar o número ótimo de clusters
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"Número ótimo de clusters determinado: {optimal_k}")
print(f"Silhouette Score para k={optimal_k}: {silhouette_scores[np.argmax(silhouette_scores)]:.4f}")
print("\n" + "="*50)

# Treinar o modelo com o número ótimo de clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Adicionar os clusters ao dataframe original
df['cluster'] = clusters

print(f"Modelo treinado com {optimal_k} clusters")
print("\nDistribuição dos pacientes por cluster:")
print(df['cluster'].value_counts().sort_index())
print("\n" + "="*50)

def describe_clusters(df, features, optimal_k):
    """
    Descreve as características de cada cluster
    """
    print("DESCRIÇÃO DOS CLUSTERS")
    print("="*60)
    
    cluster_stats = df.groupby('cluster')[features].mean()
    
    for cluster in range(optimal_k):
        print(f"\nCLUSTER {cluster}:")
        print(f"Número de pacientes: {len(df[df['cluster'] == cluster])}")
        print("-" * 40)
        
        cluster_data = df[df['cluster'] == cluster]
        
        # Características principais do cluster
        print("Características principais (valores médios):")
        for feature in features:
            mean_val = cluster_data[feature].mean()
            std_val = cluster_data[feature].std()
            print(f"  {feature}: {mean_val:.2f} ± {std_val:.2f}")
        
        # Taxa de mortalidade no cluster
        if 'DEATH_EVENT' in df.columns:
            mortality_rate = cluster_data['DEATH_EVENT'].mean() * 100
            print(f"\nTaxa de mortalidade no cluster: {mortality_rate:.1f}%")
        
        print("\n" + "-" * 40)

# Descrever os clusters
describe_clusters(df, features_for_clustering, optimal_k)

# Visualização dos clusters em 2D (usando PCA para redução de dimensionalidade)
from sklearn.decomposition import PCA

# Reduzir para 2D para visualização
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Criar dataframe para visualização
viz_df = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'cluster': clusters
})

# Plotar clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(viz_df['PC1'], viz_df['PC2'], c=viz_df['cluster'], 
                     cmap='viridis', alpha=0.7, s=60)
plt.colorbar(scatter)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variância)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variância)')
plt.title('Visualização dos Clusters (PCA)')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Variância explicada pelos componentes PCA: {pca.explained_variance_ratio_.sum():.2%}")
print("\n" + "="*50)

# Calcular médias por cluster para visualização
cluster_means = df.groupby('cluster')[features_for_clustering].mean()

# Plotar heatmap
plt.figure(figsize=(15, 8))
sns.heatmap(cluster_means.T, annot=True, cmap='YlOrRd', fmt='.2f', 
            cbar_kws={'label': 'Valor Médio'})
plt.title('Características Médias por Cluster (Valores Normalizados)')
plt.xlabel('Cluster')
plt.ylabel('Característica')
plt.tight_layout()
plt.show()

def classify_new_patient(patient_data, kmeans_model, scaler, feature_names):
    """
    Classifica um novo paciente em um cluster
    
    Parameters:
    patient_data: dict ou array com os dados do paciente
    kmeans_model: modelo KMeans treinado
    scaler: scaler treinado
    feature_names: lista com nomes das features
    
    Returns:
    cluster: cluster atribuído
    """
    
    # Converter para array numpy se for dicionário
    if isinstance(patient_data, dict):
        patient_array = np.array([patient_data[feature] for feature in feature_names]).reshape(1, -1)
    else:
        patient_array = patient_data.reshape(1, -1)
    
    # Normalizar os dados
    patient_scaled = scaler.transform(patient_array)
    
    # Prever o cluster
    cluster = kmeans_model.predict(patient_scaled)[0]
    
    return cluster

# Exemplo de uso com um novo paciente
print("EXEMPLO DE CLASSIFICAÇÃO DE NOVO PACIENTE")
print("="*50)

# Criar um paciente de exemplo (valores médios do dataset)
new_patient_example = {
    'age': 60,
    'anaemia': 0,
    'creatinine_phosphokinase': 580,
    'diabetes': 0,
    'ejection_fraction': 38,
    'high_blood_pressure': 0,
    'platelets': 263000,
    'serum_creatinine': 1.1,
    'serum_sodium': 137,
    'sex': 1,
    'smoking': 0,
    'time': 130
}

print("Dados do novo paciente:")
for key, value in new_patient_example.items():
    print(f"  {key}: {value}")

# Classificar o paciente
predicted_cluster = classify_new_patient(new_patient_example, kmeans, scaler, features_for_clustering)

print(f"\nO novo paciente foi classificado no Cluster: {predicted_cluster}")

# Mostrar características do cluster atribuído
cluster_data = df[df['cluster'] == predicted_cluster]
print(f"\nCaracterísticas do Cluster {predicted_cluster}:")
print(f"- Número de pacientes similares: {len(cluster_data)}")
print(f"- Idade média: {cluster_data['age'].mean():.1f} anos")
print(f"- Fração de ejeção média: {cluster_data['ejection_fraction'].mean():.1f}%")
print(f"- Creatinina sérica média: {cluster_data['serum_creatinine'].mean():.2f} mg/dL")

if 'DEATH_EVENT' in df.columns:
    mortality_rate = cluster_data['DEATH_EVENT'].mean() * 100
    print(f"- Taxa de mortalidade histórica: {mortality_rate:.1f}%")

    # Análise comparativa entre clusters
print("ANÁLISE COMPARATIVA ENTRE CLUSTERS")
print("="*60)

# Estatísticas por cluster
cluster_comparison = df.groupby('cluster').agg({
    'age': ['mean', 'std'],
    'ejection_fraction': ['mean', 'std'],
    'serum_creatinine': ['mean', 'std'],
    'serum_sodium': ['mean', 'std'],
    'DEATH_EVENT': 'mean' if 'DEATH_EVENT' in df.columns else 'count'
}).round(2)

print(cluster_comparison)

# Visualização das diferenças entre clusters para features importantes
important_features = ['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for i, feature in enumerate(important_features):
    df.boxplot(column=feature, by='cluster', ax=axes[i])
    axes[i].set_title(f'Distribuição de {feature} por Cluster')
    axes[i].set_ylabel(feature)

plt.suptitle('Comparação de Características Importantes entre Clusters')
plt.tight_layout()
plt.show()

print("RESUMO EXECUTIVO DA ANÁLISE DE CLUSTERS")
print("="*60)
print(f"📊 Total de pacientes analisados: {len(df)}")
print(f"🎯 Número ótimo de clusters identificado: {optimal_k}")
print(f"📈 Silhouette Score do modelo: {silhouette_score(X_scaled, clusters):.4f}")
print("\n" + "="*60)

print("\nCARACTERIZAÇÃO DOS CLUSTERS:")
for cluster in range(optimal_k):
    cluster_data = df[df['cluster'] == cluster]
    size = len(cluster_data)
    percentage = (size / len(df)) * 100
    
    print(f"\n🔹 CLUSTER {cluster} ({size} pacientes, {percentage:.1f}%):")
    
    # Identificar características distintivas
    age_mean = cluster_data['age'].mean()
    ejection_mean = cluster_data['ejection_fraction'].mean()
    creatinine_mean = cluster_data['serum_creatinine'].mean()
    
    print(f"   • Idade média: {age_mean:.1f} anos")
    print(f"   • Fração de ejeção média: {ejection_mean:.1f}%")
    print(f"   • Creatinina sérica média: {creatinine_mean:.2f} mg/dL")
    
    if 'DEATH_EVENT' in df.columns:
        mortality = cluster_data['DEATH_EVENT'].mean() * 100
        print(f"   • Taxa de mortalidade: {mortality:.1f}%")

print("\n" + "="*60)
print("🎯 APLICAÇÃO PRÁTICA:")
print("O modelo pode ser usado para:")
print("• Classificar novos pacientes em grupos de risco")
print("• Personalizar planos de tratamento")
print("• Identificar padrões clínicos específicos")
print("• Otimizar alocação de recursos hospitalares")

import joblib
import json

# Salvar o modelo treinado e o scaler
model_artifacts = {
    'kmeans_model': kmeans,
    'scaler': scaler,
    'feature_names': features_for_clustering,
    'optimal_k': optimal_k
}

joblib.dump(model_artifacts, 'heart_failure_clustering_model.pkl')

# Salvar estatísticas dos clusters
cluster_summary = df.groupby('cluster')[features_for_clustering].mean().to_dict()
with open('cluster_summary.json', 'w') as f:
    json.dump(cluster_summary, f, indent=2)

print("✅ Modelo e artefatos salvos com sucesso!")
print("📁 Arquivos gerados:")
print("   - heart_failure_clustering_model.pkl (modelo treinado)")
print("   - cluster_summary.json (estatísticas dos clusters)")

# Dados do novo paciente
new_patient = {
    'age': 65,
    'anaemia': 1,
    'creatinine_phosphokinase': 500,
    'diabetes': 1,
    'ejection_fraction': 35,
    'high_blood_pressure': 1,
    'platelets': 250000,
    'serum_creatinine': 1.4,
    'serum_sodium': 135,
    'sex': 1,
    'smoking': 1,
    'time': 100
}

cluster = classify_new_patient(new_patient, kmeans, scaler, features_for_clustering)
print(f"Paciente classificado no Cluster {cluster}")