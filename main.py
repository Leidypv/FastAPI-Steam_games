from fastapi import FastAPI
import pandas as pd
import nltk
from nltk import FreqDist
from nltk.probability import FreqDist
#from fastapi import FastAPI
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI() # Ahora la instanciamos la clase.  Ya tengo creada la aplicacion

#http://127.0.0.1:8000  # Ruta raiz del puerto
# @app.get("/") # Tenemos el objeto y la ("/") es la ruta raiz. Ejecuta la funcion

games = pd.read_csv('steam_games (original).csv')

@app.get('/genre/')
def genre(año: str):
    '''Se ingresa un año y devuelve un diccionario con los 5 géneros únicos con mayor lanzamiento por año en el orden correspondiente.'''
    
    # Filtrar los videojuegos del año ingresado
    videojuegos_año = games[games['release_year'] == int(año)]
    # Crear una copia del DataFrame para evitar modificar el original
    videojuegos_año = videojuegos_año.copy()
    
    # Descomponer la columna en filas separadas y limpiar los corchetes
    videojuegos_año['genre'] = videojuegos_año['genre'].str.replace('[', '').str.replace(']', '').str.split(',')
    # Utilizar explode para descomponer la columna en filas separadas
    generos = videojuegos_año.explode('genre')['genre']
    # Obtener los 5 géneros únicos con mayor lanzamiento por año en un diccionario
    top_5 = generos.value_counts().head(5).to_dict()

    return {f'Top 5 de géneros con mayor lanzamiento en el año {año}': top_5} 

@app.get('/title/')
def title(año: str):
    '''Se ingresa un año y devuelve un diccionario con los juegos lanzados en el año.'''

    # Filtrar los videojuegos del año ingresado
    videojuegos_año = games[games['release_year'] == int(año)]
    # Crea un diccionario con los títulos de los juegos lanzados en el año
    juegos_lanzados = videojuegos_año['title'].to_dict()

    return {f'Juegos lanzados en el año {año}': juegos_lanzados} 

@app.get('/specs/')
def specs(año: int):
    '''Se ingresa un año y devuelve un diccionario con los 5 specs que más se repiten en el mismo en el orden correspondiente.'''

    # Filtrar los videojuegos del año ingresado
    videojuegos_año = games[games['release_year'] == int(año)]
    # "Explotar" la columna para obtener una fila por cada spec en una lista separada
    explode = videojuegos_año.explode('specs')
    # Calcular la frecuencia utilizando nltk.FreqDist
    freq_dist = FreqDist(explode['specs'])
    # Obtener los 5 más frecuentes
    top_5 = freq_dist.most_common(5)
    # Convertir el resultado en un diccionario
    cantidad = dict(top_5)

    return {f'Top 5 de specs que más se repiten en el año {año}': cantidad}

@app.get('/earlyacces/')
def earlyacces(año: str):
    ''' Cantidad de juegos lanzados en un año con early access.'''
    
    # Filtrar los videojuegos del año ingresado y que esten en la columna "early access"
    videojuegos = games[(games['release_year'] == int(año)) & (games['early_access'] == True)]
    # Contar la cantidad de juegos con "early access"
    cantidad = len(videojuegos)

    return {f'Cantidad de juegos lanzados en el año {año} con early access': cantidad}

@app.get('/sentiment/')
def sentiment(año: str):
    ''' Según el año de lanzamiento, se devuelve un diccionario con la cantidad de registros categorizados con un análisis de sentimiento.'''
    
    # Filtrar los videojuegos del año ingresado
    videojuegos = games[games['release_year'] == int(año)]
    # Contar la cantidad de registros para cada categoría de sentimiento
    cantidad = videojuegos['sentiment'].value_counts()

    return cantidad.to_dict()

@app.get('/metascore/')
def metascore(año: str):
    ''' Top 5 de juegos según año con mayor metascore.'''
    
    # Filtrar los videojuegos del año ingresado
    videojuegos = games[games['release_year'] == int(año)]
    # Ordenar los videojuegos por metascore de mayor a menor
    videojuegos = videojuegos.sort_values(by='metascore', ascending=False)
    # Tomar los 5 primeros videojuegos del DataFrame ordenado
    top_5 = videojuegos.head(5)
    
    return {f'Top 5 de juegos del año {año} con mayor metascore': top_5.set_index('title')['metascore'].to_dict()}


games = pd.read_csv('steam_games (original).csv', usecols=['tags','title','release_year','metascore','price','discount_price'])
games['tags'] = games['tags'].str.replace('[', '').str.replace(']', '').str.replace("'", "").str.lower()

# Crear matriz TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(games['tags'])

@app.get("/predicción/", response_model=List[dict])
def predicción(tags_query: str):
    ''' Segun el tags(etiqueta), le devuelve un diccionario con el title del videojuego,year,price,metascore y discount.'''

    # Convertir la consulta a minúsculas
    tags_query = tags_query.lower()

    # Transformar la consulta a una matriz de características utilizando TF-IDF
    query_vector = tfidf_vectorizer.transform([tags_query])

    # Calcular similitud coseno entre la consulta y los datos
    similarity = cosine_similarity(query_vector, tfidf_matrix)

    # Encontrar los índices de los juegos más similares
    similar_indices = similarity.argsort()[0][-5:][::-1]

    # Obtener la información de los juegos más similares
    similar_games = games.iloc[similar_indices]

    # Crear una lista con la información de los juegos similares
    games_info = []
    for _, row in similar_games.iterrows():
        game_info = {
            'title': row['title'],
            'release_year': row['release_year'],
            'price': row['price'],
            'metascore': row['metascore'],
            'discount_price': row['discount_price'],
        }
        games_info.append(game_info)
    
    return games_info, {f'RMSE del modelo: 13.927227354733178'}




