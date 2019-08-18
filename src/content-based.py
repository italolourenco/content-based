import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def main():

    ratingsSD = "../dataset/SmallData/ratings.csv"
    moviesSD = "../dataset/SmallData/movies.csv"

    df_ratings = pd.read_csv(ratingsSD, usecols=['userId', 'movieId', 'rating'])
    df_movies = pd.read_csv(moviesSD, usecols=['movieId', 'title', 'genres'])

    dataset = pd.merge(df_movies, df_ratings, on='movieId', how='inner')

    df_movies['genres'] = df_movies['genres'].str.split('|')

    df_movies['genres'] = df_movies['genres'].fillna("").astype('str')

    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(df_movies['genres'])
    
    print tfidf_matrix

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    titles = df_movies['title']
    indices = pd.Series(df_movies.index, index=df_movies['title'])

    title = 'Good Will Hunting (1997)'

    idx = indices[title]
    print idx
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]

    print movie_indices

    print titles.iloc[movie_indices]



if __name__ == '__main__':
	main()