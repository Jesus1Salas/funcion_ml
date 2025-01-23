# 1. Importar libreria para el modelo de ML
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD



# 2. Importar libreria para vincular con Big Query
# Esta parte del código lo dejo para cuando te vincules a Big Query. Yo voy a usar un modelo 3 que te voy a pasar


# 3. Descargar el dataset (yo voy a trabajar en local, pero esto habría que cambiarlo para que se vincule directamente con Big Query)
df_ml_3 = pd.read_parquet('Sisreco\modelo3.parquet')

# 4. Funcion 1:

# Eliminamos las filas duplicadas excepto el id_user
df_ml_3_1 = df_ml_3.drop_duplicates(
    subset=['id_business', 'business_name', 'city_name', 'business_address'])

# Filtrar filas correspondientes al año 2021
df_ml_3_1 = df_ml_3_1[(df_ml_3['date'].dt.year == 2021)]

# Check for 'description' column existence and fill NaN with an empty string
if 'description' in df_ml_3_1.columns:
    df_ml_3_1['description'] = df_ml_3_1['description'].fillna('')
else:
    raise ValueError("The DataFrame does not contain a 'description' column.")

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')

# Fit and transform the data
tfidf_matrix = vectorizer.fit_transform(df_ml_3_1['description'])


def get_recommendations(keyword, city=None, avg_rating=None, business_name=None, region=None, df=df_ml_3_1, vectorizer=vectorizer, svd=None):
    filtered_df = df.copy()
    if city:
        filtered_df = filtered_df[filtered_df['city_name'] == city]
    if avg_rating:
        filtered_df = filtered_df[filtered_df['avg_rating'] >= avg_rating]
    if business_name:
        filtered_df = filtered_df[filtered_df['business_name'].str.contains(
            business_name, case=False)]
    if region:
        filtered_df = filtered_df[filtered_df['region'].str.contains(
            region, case=False)]

    # Handle empty DataFrame
    if filtered_df.empty:
        return pd.DataFrame({'business_name': [], 'similarity': [], 'city_name': [], 'avg_rating': [], 'region': [], 'weighted_rating': []})

    # Drop duplicates and handle NaN descriptions
    filtered_df = filtered_df.drop_duplicates(subset=['id_business'])
    filtered_df['description'] = filtered_df['description'].fillna('')

    tfidf_matrix_filtered = vectorizer.transform(filtered_df['description'])

    # Ensure n_components is within range of features
    n_features = tfidf_matrix_filtered.shape[1]
    n_components = min(100, n_features)

    if svd:
        svd.n_components = n_components
        tfidf_matrix_filtered = svd.fit_transform(tfidf_matrix_filtered)

    keyword_tfidf = vectorizer.transform([keyword])
    if svd:
        keyword_tfidf = svd.transform(keyword_tfidf)

    keyword_sim = cosine_similarity(keyword_tfidf, tfidf_matrix_filtered)
    sim_scores = list(enumerate(keyword_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    filtered_scores = [score for score in sim_scores if score[1] > 0.1]

    top_indices = [i[0] for i in filtered_scores[:3]]
    scores = [i[1] for i in filtered_scores[:3]]

    # Round similarity percentages to the nearest integer
    similarity_percentages = [f"{round(score * 100)}%" for score in scores]

    recommendations = pd.DataFrame({
        'business_name': filtered_df['business_name'].iloc[top_indices].values,
        'category_name': filtered_df['category_name'].iloc[top_indices].values,
        'similarity': similarity_percentages,
        'city_name': filtered_df['city_name'].iloc[top_indices].values,
        'business_address': filtered_df['business_address'].iloc[top_indices].values,
        'avg_rating': filtered_df['avg_rating'].iloc[top_indices].values,
        'region': filtered_df['region'].iloc[top_indices].values,
        'hours': filtered_df['hours'].iloc[top_indices].values,
        'weighted_rating': filtered_df['weighted_rating'].iloc[top_indices].values
    })

    # Ordenar el DataFrame resultante por la columna 'weighted_rating' en orden descendente
    recommendations = recommendations.sort_values(
        by='weighted_rating', ascending=False)

    # Eliminar la columna 'weighted_rating' del resultado final
    recommendations = recommendations.drop(columns=['weighted_rating'])

    return recommendations


# Ejemplo de llamada a la función
recommended = get_recommendations('Tomato', svd=TruncatedSVD(n_components=100))
print("Recommended Restaurants:")
print(recommended)
