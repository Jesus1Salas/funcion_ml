from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import io
from google.cloud import storage
from google.auth import default
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Inicializar la aplicación FastAPI
app = FastAPI()

# Inicializar cliente de Google Cloud
credentials, project = default()
storage_client = storage.Client(credentials=credentials)

# Definir el bucket y el dataset
bucket_name = "datosfuentes"
file_name="tablas/modelo3.parquet"

# Cargar el dataset
try:
    #df_ml_3 = pd.read_parquet('Sisreco/modelo3.parquet')
    bucket = storage_client.get_bucket(bucket_name)
    blob=bucket.blob(file_name)
    file_contents = blob.download_as_string()
    df_ml_3 = pd.read_parquet(io.BytesIO(file_contents))

except Exception as e:
    raise RuntimeError(f"Error loading dataset: {e}")

# Preprocesamiento inicial
df_ml_3_1 = df_ml_3.drop_duplicates(
    subset=['id_business', 'business_name', 'city_name', 'business_address']
)
df_ml_3_1 = df_ml_3_1[(df_ml_3['date'].dt.year == 2021)]

df_ml_3_1['description'] = df_ml_3_1.get('description', pd.Series()).fillna('')

# Inicializar el vectorizador TF-IDF
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df_ml_3_1['description'])

# Definir modelo de entrada para la solicitud
class RecommendationRequest(BaseModel):
    keyword: str
    city: Optional[str] = None
    avg_rating: Optional[float] = None
    business_name: Optional[str] = None
    region: Optional[str] = None

# Función para obtener recomendaciones
def get_recommendations(keyword, city=None, avg_rating=None, business_name=None, region=None, df=df_ml_3_1, vectorizer=vectorizer, svd=None):
    filtered_df = df.copy()
    if city:
        filtered_df = filtered_df[filtered_df['city_name'] == city]
    if avg_rating:
        filtered_df = filtered_df[filtered_df['avg_rating'] >= avg_rating]
    if business_name:
        filtered_df = filtered_df[filtered_df['business_name'].str.contains(business_name, case=False)]
    if region:
        filtered_df = filtered_df[filtered_df['region'].str.contains(region, case=False)]

    if filtered_df.empty:
        return []

    filtered_df = filtered_df.drop_duplicates(subset=['id_business'])
    filtered_df['description'] = filtered_df['description'].fillna('')

    tfidf_matrix_filtered = vectorizer.transform(filtered_df['description'])

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

    recommendations = recommendations.sort_values(by='weighted_rating', ascending=False)
    recommendations = recommendations.drop(columns=['weighted_rating'])

    return recommendations.to_dict(orient='records')

# Endpoint para obtener recomendaciones
@app.post("/recommendations")
def recommend(request: RecommendationRequest):
    svd = TruncatedSVD(n_components=100)
    try:
        recommendations = get_recommendations(
            keyword=request.keyword,
            city=request.city,
            avg_rating=request.avg_rating,
            business_name=request.business_name,
            region=request.region,
            svd=svd
        )
        return {"recommendations": recommendations}
    except Exception as e:
        return {"error": str(e)}