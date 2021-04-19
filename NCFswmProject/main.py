import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import NCFModel
import pytorch_lightning as pl
from pprint import pprint
from flask import Flask
from flask_restful import Resource, Api, reqparse
import ast

np.random.seed(123)

movies = pd.read_csv('../movies.csv')
ratings = pd.read_csv('../ratings.csv', parse_dates=['timestamp'], nrows=5000000)
ratings['timestamp'] = ratings['timestamp'].astype(int)
ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
test_ratings = ratings[ratings['rank_latest'] == 1]
test_ratings = test_ratings[['userId', 'movieId', 'rating']]
all_movieIds = ratings['movieId'].unique()
# Dict of all items that are interacted with by each user
user_interacted_items = ratings.groupby('userId')['movieId'].apply(list).to_dict()
model = torch.load("trained.model")

def get_movie_title_from_id(movie_id):
    return movies[movies['movieId'] == movie_id].iloc[0]['title']

def predict(user_id):
    np.random.seed(123)
    # test_ratings = test_ratings[177:178]
    filtered_test_ratings = test_ratings.loc[test_ratings["userId"] == user_id]

    # User-item pairs for testing
    test_user_item_set = set(zip(filtered_test_ratings['userId'], filtered_test_ratings['movieId']))

    hits = []
    user_and_item = list(test_user_item_set)[0]
    u, i = user_and_item
    interacted_items = user_interacted_items[u]
    not_interacted_items = set(all_movieIds) - set(interacted_items)
    selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
    test_items = selected_not_interacted + [i]

    predicted_labels = np.squeeze(model(torch.tensor([u] * 100),
                                        torch.tensor(test_items)).detach().numpy())
    sorted_score_movieid_pairs = sorted(zip(predicted_labels, test_items), reverse=True)
    top_10_sorted_score_movieid_pairs = sorted_score_movieid_pairs[:10]
    top10_items = [m for s,m in top_10_sorted_score_movieid_pairs]
    print("User interacted with Movie id: ", i)
    print("Movie recommendations for User id: ", u)
    print(top10_items)
    if i in top10_items:
        hits.append(1)
    else:
        hits.append(0)

    hit_ratio = np.average(hits)
    print("The Hit Ratio @ 10 is {:.2f}".format(hit_ratio))

    return {
        "user_id": str(u),
        "latest_interacted_movie": str(i),
        "scores": [("%.4f" % s) for s,m in sorted_score_movieid_pairs[:10]],
        "movie_ids": [str(m) for s,m in sorted_score_movieid_pairs[:10]],
        "movie_names": [get_movie_title_from_id(m) for s,m in top_10_sorted_score_movieid_pairs],
        "hit_ratio": "%.4f" % hit_ratio
    }

class User(Resource):
    def get(self):
        parser = reqparse.RequestParser()  # initialize

        parser.add_argument('userId', required=True)  # add args

        args = parser.parse_args()  # parse arguments to dictionary

        a = int(args['userId'])
        print(a)
        print(type(a))
        # return predict(a)
        return predict(a)


if __name__ == "__main__":
    USER_ID = 178

    predictions = predict(178)
    pprint(predictions)

    predictions = predict(178)
    pprint(predictions)

    predictions = predict(178)
    pprint(predictions)



    app = Flask(__name__)
    api = Api(app)

    api.add_resource(User, '/users')  # '/users' is our entry point

    app.run()
