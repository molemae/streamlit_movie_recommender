"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""
# import
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from utils import ratings

# open NMF model:
with open('data/nmf_model2.pkl',mode='rb') as openfile:
   nmf_model = pickle.load(openfile)


def recommend_random(k=10):
    #return pd.DataFrame(ratings.index).sample(k)['userId'].to_list()
    return list(pd.DataFrame(ratings.columns).sample(k)[0])

def recommend_with_NMF(query, k=10):
    """
    NMF Recommender
    INPUT
    - user_vector with shape (1, #number of movies)
    - user_item_matrix
    - trained NMF model

    OUTPUT
    - a list of movieIds
    """
    df = pd.DataFrame(query, columns=list(ratings.columns), index=["new_user"])
    df = df.fillna(ratings.mean())
    P_new_user_matrix = nmf_model.transform(df)
    R_hat_new_user_matrix = np.dot(P_new_user_matrix, nmf_model.components_)
    R_hat_new_user = pd.DataFrame(data=R_hat_new_user_matrix,
                            columns=ratings.columns,
                            index = ['new_user'])

    sorted_list = R_hat_new_user.transpose().sort_values(by="new_user", ascending=False).index.to_list()
    rated_movies = list(query.keys())
    recommended_movies = [movie for movie in sorted_list if movie not in rated_movies]
    recommended_movies = recommended_movies[:k] #shorten to Top 10
    return recommended_movies

def recommend_neighborhood(query,k=5):
    # transpose ratings matrix
    initial = ratings.T
    # add new query to user item matrix
    query_df = pd.DataFrame(query.values(), index=query.keys(),columns=['new_user'])
    user_item = initial.merge(query_df,how='left',left_index=True,right_index=True)
    
    # unseen movies
    unseen_movies = user_item[user_item['new_user'].isna()].index
    
    # fill na for cosine similarity
    user_item = user_item.fillna(0)

    # create cosine similarity matrix
    user_user = cosine_similarity(user_item.T)
    user_user = pd.DataFrame(user_user, columns = user_item.columns, index = user_item.columns).round(2)

     # select 5 users with highest similarity
    top_five_users = user_user['new_user'].sort_values(ascending=False).index[1:6]

    movie_rec_list = list()
    ratio_list = list()
    for movie in unseen_movies:
        other_users = initial.columns[~initial.loc[movie].isna()]
        other_users = set(other_users)

        num = 0
        den = 0
        for other_user in other_users.intersection(set(top_five_users)):
            rating = user_item[other_user][movie]
            sim = user_user['new_user'][other_user]
            num = num + (rating*sim)
            den = den + sim + 0.0001
            ratio = num/den
        movie_rec_list.append(movie)
        ratio_list.append(ratio)
    out = pd.DataFrame({
        'movie':movie_rec_list,
        'ratio':ratio_list
    })
    out.sort_values('ratio',ascending=False,inplace=True)
    
    return list(out.iloc[:k]['movie']) #movie_rec_list, ratio_list

    

