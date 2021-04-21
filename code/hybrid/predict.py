import pickle
import pandas as pd
from flask import Flask
from flask_restful import Resource, Api, reqparse

ratingsdata= pd.read_csv('../../data/ratings.csv')

ratingsdata=ratingsdata.sample(n=5000000)
filtered_ratings=ratingsdata.groupby('userId').filter(lambda x:len(x)>= 30)
mlr = filtered_ratings.movieId.unique().tolist()
filtered_ratings.drop(['timestamp'],1, inplace=True)

moviesdata= pd.read_csv('../../data/movies.csv')
moviesdata['genres']=moviesdata['genres'].str.replace('|',' ')
moviesdata=moviesdata[moviesdata.movieId.isin(mlr)]

Mapit = dict(zip(moviesdata.title.tolist(),moviesdata.movieId.tolist()))

algo2 = pickle.load(open("hybrid.model", 'rb'))

def recommovie2(ui):
    if ui in filtered_ratings.userId.unique():
        listforuser = filtered_ratings[filtered_ratings.userId == ui].movieId.tolist()
        d = {k: v for k, v in Mapit.items() if not v in listforuser}
        predlus = []
        for i, j in d.items():
            predic = algo2.predict(ui, j)
            predlus.append((i, predic[3]))
        predf = pd.DataFrame(predlus, columns=['movies', 'ratings'])
        predf.sort_values('ratings', ascending=False, inplace=True)
        predf.set_index('movies', inplace=True)
        return predf.head(10)
    else:
        print("Does not exist")
        return None

# print(recommovie2(33316))

class HybridUser(Resource):
    def get(self):
        parser = reqparse.RequestParser()  # initialize
        parser.add_argument('userId', required=True)  # add args
        args = parser.parse_args()  # parse arguments to dictionary
        user_id = int(args['userId'])
        df = recommovie2(user_id)
        return {
            "user_id": user_id,
            "scores": [("%.4f" % rating) for rating in df['ratings'].values],
            "movie_names": [title for title in df['ratings'].index]
        }

class AllHybridUsers(Resource):
    def get(self):
        uusers = [str(user_id) for user_id in filtered_ratings.userId.unique()[:50]]
        return uusers