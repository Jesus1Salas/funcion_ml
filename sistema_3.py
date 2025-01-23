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

# 4. Funcion 3:

# Función para realizar los filtros iniciales según la ciudad y calificación mínima


def filter_df(df, city_name=None, min_rating=4):
    filtered_df = df.copy()
    if city_name:
        filtered_df = filtered_df[filtered_df['city_name'] == city_name]
    filtered_df = filtered_df[filtered_df['avg_rating'] >= min_rating]
    return filtered_df


# Aplicar los filtros dinámicamente
# Cambia este valor si deseas otra ciudad, None para no filtrarlo
city_name = "Philadelphia"

min_rating = 4  # Cambia este valor si deseas otro rating mínimo

df_ml_3_3 = filter_df(df_ml_3, city_name=city_name, min_rating=min_rating)

# Filtrar un subconjunto de usuarios más activos
active_users = df_ml_3_3['id_user'].value_counts().head(
    100).index.tolist()  # Convertir a lista
df_ml_3_3 = df_ml_3_3[df_ml_3_3['id_user'].isin(active_users)]

# Filtrar solo las reseñas de 4 o 5 estrellas
df_ml_3_3['avg_rating_binary'] = df_ml_3_3['avg_rating'].apply(
    lambda x: 1 if x >= 4 else 0)

# Crear una matriz de usuario-restaurante basada en las reseñas de 4 o 5 estrellas
user_restaurant_matrix = df_ml_3_3.pivot_table(
    index='id_user', columns='business_name', values='avg_rating_binary')
user_restaurant_matrix.fillna(0, inplace=True)

# Reducir la dimensionalidad con TruncatedSVD
svd = TruncatedSVD(n_components=100)
user_restaurant_matrix_reduced = svd.fit_transform(user_restaurant_matrix)

# Calcular la similitud entre usuarios usando la matriz reducida
user_similarity = cosine_similarity(user_restaurant_matrix_reduced)
user_similarity_df = pd.DataFrame(
    user_similarity, index=user_restaurant_matrix.index, columns=user_restaurant_matrix.index)

# Separar en conjunto de entrenamiento y prueba
train_df, test_df = train_test_split(df_ml_3_3, test_size=0.2, random_state=42)

# Crear la matriz usuario-restaurante para entrenamiento
train_user_restaurant_matrix = train_df.pivot_table(
    index='id_user', columns='business_name', values='avg_rating_binary')
train_user_restaurant_matrix.fillna(0, inplace=True)

# Función de recomendaciones colaborativas


def get_recommendations(id_user, n_recommendations=3, user_restaurant_matrix=train_user_restaurant_matrix, user_similarity_df=user_similarity_df, sample_df=train_df):
    avg_ratings = user_restaurant_matrix.loc[id_user]
    similar_users = user_similarity_df[id_user].sort_values(ascending=False)
    weighted_ratings = user_restaurant_matrix.T.dot(similar_users)

    scaler = MinMaxScaler()
    weighted_ratings = pd.Series(scaler.fit_transform(
        weighted_ratings.values.reshape(-1, 1)).flatten(), index=weighted_ratings.index)

    rated_restaurants = avg_ratings[avg_ratings == 1].index
    weighted_ratings.drop(rated_restaurants, inplace=True)

    top_recommendations = weighted_ratings.sort_values(
        ascending=False).head(n_recommendations)
    # Crear el DataFrame de recomendaciones con el mismo índice que top_recommendations
    recommendations = sample_df[sample_df['business_name'].isin(
        top_recommendations.index)].copy()

    # Emparejar y asignar las similitudes correctas
    similarity_dict = top_recommendations.to_dict()
    recommendations['similarity'] = recommendations['business_name'].map(
        similarity_dict)
    recommendations['similarity'] = recommendations['similarity'].apply(
        lambda x: "{:.2f}%".format(x * 100))

    # Ordenar por weighted_rating en orden descendente
    recommendations.sort_values(
        by='weighted_rating', ascending=False, inplace=True)
    # Eliminar duplicados basados en 'business_name' y 'business_address'
    recommendations.drop_duplicates(
        subset=['business_name', 'business_address'], inplace=True)
    # Ajustar los índices después del drop duplicates
    recommendations.reset_index(drop=True, inplace=True)

    # Seleccionar y devolver solo las columnas especificadas en el orden solicitado, omitiendo 'weighted_rating'
    return recommendations[['business_name', 'category_name', 'similarity', 'city_name', 'business_address', 'avg_rating', 'region', 'hours']].head(n_recommendations)


# Seleccionar un id_user aleatorio del DataFrame de entrenamiento
random_user = random.choice(train_df['id_user'].unique())

# Obtener recomendaciones para el usuario aleatorio
recommendations = get_recommendations(random_user, n_recommendations=3)

# Mostrar las tres primeras recomendaciones
print(
    f"Recommended Restaurants for user {random_user} in {city_name} with min rating {min_rating}:{recommendations}")

