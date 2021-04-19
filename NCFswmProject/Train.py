import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import NCFModel
import pytorch_lightning as pl

np.random.seed(123)

def main():
    ratings = pd.read_csv('../ratings.csv', parse_dates=['timestamp'], nrows=5000000)

    # train test split starts here
    ratings['timestamp'] = ratings['timestamp'].astype(int)

    ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)

    train_ratings = ratings[ratings['rank_latest'] != 1]

    # drop columns that we no longer need
    train_ratings = train_ratings[['userId', 'movieId', 'rating']]

    # convert to implicit dataset from explicit

    train_ratings.loc[:, 'rating'] = 1

    # ********* Negative sampling starts here **********

    # Get a list of all movie IDs
    all_movieIds = ratings['movieId'].unique()

    # Placeholders that will hold the training data
    users, items, labels = [], [], []

    # This is the set of items that each user has interaction with
    user_item_set = set(zip(train_ratings['userId'], train_ratings['movieId']))

    # 4:1 ratio of negative to positive samples
    num_negatives = 4

    for (u, i) in tqdm(user_item_set):
        users.append(u)
        items.append(i)
        labels.append(1)  # items that the user has interacted with are positive
        for _ in range(num_negatives):
            # randomly select an item
            negative_item = np.random.choice(all_movieIds)
            # check that the user has not interacted with this item
            while (u, negative_item) in user_item_set:
                negative_item = np.random.choice(all_movieIds)
            users.append(u)
            items.append(negative_item)
            labels.append(0)  # items not interacted with are negative

    # ********* training ************

    num_users = ratings['userId'].max() + 1
    num_items = ratings['movieId'].max() + 1
    all_movieIds = ratings['movieId'].unique()

    model = NCFModel.NCF(num_users, num_items, train_ratings, all_movieIds)

    trainer = pl.Trainer(max_epochs=5, gpus=1, reload_dataloaders_every_epoch=True,
                         progress_bar_refresh_rate=50, logger=False, checkpoint_callback=False)

    trainer.fit(model)

    torch.save(model, "trained.model")


if __name__ == "__main__":
    main()
