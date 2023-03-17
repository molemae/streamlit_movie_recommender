"""
UTILS 
- Helper functions to use for your recommender funcions, etc
- Data: import files/models here e.g.
    - movies: list of movie titles and assigned cluster
    - ratings
    - user_item_matrix
    - item-item matrix 
- Models:
    - nmf_model: trained sklearn NMF model
"""
import pandas as pd
import numpy as np

ratings = pd.read_csv('data/Ratings.csv', index_col='userId') 
movies = pd.DataFrame(zip(list(range((ratings.shape[1]))),ratings.columns),columns=['id','title'])


def movie_to_id(title, movie_and_ID=movies):
    '''
    converts movie title to id for use in algorithms'''
    
    movieID = movie_and_ID[movie_and_ID['title']==title]['id']
    
    return movieID

def id_to_movie(movieID,movie_and_ID=movies):
    '''
    converts movie Id to title
    '''
    rec_title = movie_and_ID[movie_and_ID['id']==movieID]['title']
    
    return rec_title

