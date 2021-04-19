import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import NCFModel
import pytorch_lightning as pl

np.random.seed(123)


def main():
    ratings = pd.read_csv('ratings.csv', parse_dates=['timestamp'], nrows=5000000)
    # ratings = pd.read_csv('ratings.csv', parse_dates=['timestamp'])

    # rand_userIds = np.random.choice(ratings['userId'].unique(),
    #                               size=int(len(ratings['userId'].unique()) * 0.3),
    #                               replace=False)

    # ratings = ratings.loc[ratings['userId'].isin(rand_userIds)]

    ratings['timestamp'] = ratings['timestamp'].astype(int)

    ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)

    test_ratings = ratings[ratings['rank_latest'] == 1]

    test_ratings = test_ratings[['userId', 'movieId', 'rating']]

    test_ratings = test_ratings[177:178]

    all_movieIds = ratings['movieId'].unique()

    model = torch.load("trained.model")
    # User-item pairs for testing
    test_user_item_set = set(zip(test_ratings['userId'], test_ratings['movieId']))

    # Dict of all items that are interacted with by each user
    user_interacted_items = ratings.groupby('userId')['movieId'].apply(list).to_dict()
    print(len(test_user_item_set))
    hits = []
    for (u, i) in test_user_item_set:
        interacted_items = user_interacted_items[u]
        not_interacted_items = set(all_movieIds) - set(interacted_items)
        selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
        test_items = selected_not_interacted + [i]

        predicted_labels = np.squeeze(model(torch.tensor([u] * 100),
                                            torch.tensor(test_items)).detach().numpy())
        top10_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][0:10].tolist()]
        print("User interacted with Movie id: ", i)
        print("Movie recommendations for User id: ", u)
        print(top10_items)
        if i in top10_items:
            hits.append(1)
        else:
            hits.append(0)

    print("The Hit Ratio @ 10 is {:.2f}".format(np.average(hits)))


if __name__ == "__main__":
    main()
