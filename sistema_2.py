# 1. Importar libreria para el modelo de ML
import pandas as pd
import spacy
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ndcg_score
from sklearn.metrics import ndcg_score
from sklearn.model_selection import KFold
import random

# 2. Importar libreria para vincular con Big Query
# Esta parte del código lo dejo para cuando te vincules a Big Query. Yo voy a usar un modelo 3 que te voy a pasar

# 3. Descargar el dataset (yo voy a trabajar en local, pero esto habría que cambiarlo para que se vincule directamente con Big Query)
df_ml_3 = pd.read_parquet('modelo3j.parquet')

# 4. Funcion 2:

# Definir la función de filtrado


def filter_dataframe(df, city_name=None, custom_min_rating=None):
    if city_name:
        df = df[df['city_name'] == city_name]
    if custom_min_rating:
        df = df[df['avg_rating'] >= custom_min_rating]
    return df


# Filtrar el DataFrame original antes de continuar
city_name = "Richboro"  # Puedes cambiar la ciudad según lo necesites
custom_min_rating = 4  # Cambia este valor si deseas otro rating mínimo
df_ml_3 = filter_dataframe(df_ml_3, city_name, custom_min_rating)

# Filtrar un subconjunto de usuarios más activos
active_users = df_ml_3['id_user'].value_counts().head(
    100).index.tolist()  # Convertir a lista
df_ml_3_2 = df_ml_3[df_ml_3['id_user'].isin(active_users)]

# Dividir los datos en conjunto de entrenamiento y prueba
train_df, test_df = train_test_split(
    df_ml_3_2, test_size=0.25, random_state=42)

# Verificar el tamaño de los conjuntos de datos
print(f"Tamaño del conjunto de entrenamiento: {train_df.shape}")
print(f"Tamaño del conjunto de prueba: {test_df.shape}")

# Lidiar con valores nulos en la columna 'description'
train_df['description'] = train_df['description'].fillna('')

# Reindexar el DataFrame de entrenamiento para asegurar que los índices sean consecutivos
train_df = train_df.reset_index(drop=True)

# Vectorizar las descripciones de los ítems del conjunto de entrenamiento
# Limitar el número de características
vectorizer = TfidfVectorizer(max_features=10000)
train_tfidf_matrix = vectorizer.fit_transform(train_df['description'])

# Calcular la similitud de coseno para el conjunto de entrenamiento
train_cosine_sim = cosine_similarity(train_tfidf_matrix, train_tfidf_matrix)

# Verificar que la matriz de similitud tenga el tamaño adecuado
print(f"Tamaño de la matriz de similitud: {train_cosine_sim.shape}")

# Crear un índice rápido para búsqueda
item_to_index = pd.Series(
    train_df.index, index=train_df['id_business']).to_dict()


def get_recommendations(id_user, train_df, test_df, cosine_sim, top_n=3):
    user_interactions = test_df[test_df['id_user'] == id_user]
    user_items = user_interactions['id_business'].values

    if len(user_items) == 0:
        return pd.DataFrame(columns=['business_name', 'category_name', 'similarity', 'city_name', 'business_address', 'avg_rating', 'region', 'hours'])

    scores = {}
    for item in user_items:
        if item in train_df['id_business'].values:
            item_idx = train_df[train_df['id_business'] == item].index[0]
            if item_idx >= len(cosine_sim):
                print(
                    f"Índice {item_idx} fuera de los límites de la matriz de similitud.")
                continue

            sim_scores = cosine_sim[item_idx]
            for idx, score in enumerate(sim_scores):
                if idx >= len(train_df):
                    print(
                        f"Índice {idx} fuera de los límites del DataFrame de entrenamiento.")
                    continue

                restaurant_id = train_df.iloc[idx]['id_business']
                if restaurant_id not in user_items:
                    if restaurant_id not in scores:
                        scores[restaurant_id] = score
                    else:
                        scores[restaurant_id] += score

    if not scores:
        print(f"No se encontraron recomendaciones para el usuario {id_user}.")
        return pd.DataFrame(columns=['business_name', 'category_name', 'similarity', 'city_name', 'business_address', 'avg_rating', 'region', 'hours'])

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    recommended_items = [item[0] for item in sorted_scores[:top_n]]
    scores = [max(min(item[1] * 100, 100), 0)
              for item in sorted_scores[:len(recommended_items)]]

    recommendations = train_df[train_df['id_business'].isin(
        recommended_items)].copy()
    recommendations.drop_duplicates(
        subset=['business_name', 'business_address'], inplace=True)

    scores = scores[:len(recommendations)]
    recommendations['similarity'] = [f"{score:.0f}%" for score in scores]
    recommendations.sort_values(
        by='weighted_rating', ascending=False, inplace=True)

    return recommendations[['business_name', 'category_name', 'similarity', 'city_name', 'business_address', 'avg_rating', 'region', 'hours']].head(top_n)


# Ejemplo de uso: Seleccionar un id_user aleatorio y obtener recomendaciones
random_user = random.choice(test_df['id_user'].unique())
recommendations = get_recommendations(
    random_user, train_df, test_df, train_cosine_sim)
print(
    f"Recommended Restaurants for user {random_user} in {city_name} with min rating {custom_min_rating}:{recommendations}")
recommendations
