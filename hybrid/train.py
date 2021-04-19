import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle
from matplotlib import pyplot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
#!pip install surprise
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import KFold
from collections import defaultdict

moviesdata= pd.read_csv('../movies.csv')
tagsdata= pd.read_csv('../tags.csv')
ratingsdata= pd.read_csv('../ratings.csv')

ratingsdata=ratingsdata.sample(n=5000000)
moviesdata

moviesdata['genres']=moviesdata['genres'].str.replace('|',' ')

len(moviesdata.movieId.unique())

ratingsdata

#The rating data can be shrinked to consider only the users that have rated more than 30 movies.
filtered_ratings=ratingsdata.groupby('userId').filter(lambda x:len(x)>= 30)
#This allows the data to retain most of the movies while reducing the amount of users.
mlr = filtered_ratings.movieId.unique().tolist()
titlesremaining = len(filtered_ratings.movieId.unique())/len(moviesdata.movieId.unique()) * 100

# we take only the unique movie titles that are also present in the new filtered data.
moviesdata=moviesdata[moviesdata.movieId.isin(mlr)]

moviesdata.head(10)
# so now we have the genres without the | seperators

Mapit = dict(zip(moviesdata.title.tolist(),moviesdata.movieId.tolist()))
#create a dictionary for movie titles and id for fast lookup

#getting rid of timestamp column as it is not needed
tagsdata.head
tagsdata.drop(['timestamp'],1,inplace=True)
filtered_ratings.drop(['timestamp'],1, inplace=True)

#merging the dataframes to get consolidated dataset
mxmat=pd.merge(moviesdata,tagsdata,on='movieId',how='left')
mxmat.head(10)

mxmat.fillna("",inplace=True)
mxmat = pd.DataFrame(mxmat.groupby('movieId')['tag'].apply(lambda x: "%s" % ' '.join(x)))
finmat = pd.merge(moviesdata, mxmat, on='movieId', how='left')
finmat ['metdat'] = finmat[['tag', 'genres']].apply(lambda x: ' '.join(x), axis=1)
finmat[['movieId','title','metdat']].head(10)
# combining tags and genres to generte data for content based recommendation later

print('Percentage of titles remaining in dataset out of original:')
print(titlesremaining)
termfrinvfr=TfidfVectorizer(stop_words='english')
termfrinvfr_matrix=termfrinvfr.fit_transform(finmat['metdat'])
termfrinvfr_df = pd.DataFrame(termfrinvfr_matrix.toarray(), index=finmat.index.tolist())
# using stopwords from sklearn library to generate tf-idf matrix
#print(termfrinvfr_df.shape)

singularVD = TruncatedSVD(n_components=200)
intermatgen = singularVD.fit_transform(termfrinvfr_matrix)
n = 200
intermatgen_1_df = pd.DataFrame(intermatgen[:,0:n], index=finmat.title.tolist())
intermatgen.shape

filtered_ratings.head(10)

filtered_ratings1 = pd.merge(moviesdata[['movieId']], filtered_ratings, on="movieId", how="right")

filtered_ratings1.head(10)
filtered_ratings2=pd.pivot(filtered_ratings1,index = 'movieId', columns ='userId', values = 'rating').fillna(0)

filtered_ratings2.head(10)

len(filtered_ratings.movieId.unique())

singularVD = TruncatedSVD(n_components=200)
intermatgensec = singularVD.fit_transform(filtered_ratings2)
intermatgensecdata = pd.DataFrame(intermatgensec, index=finmat.title.tolist())

explotter = singularVD.explained_variance_ratio_.cumsum()
plt.plot(explotter,'-', ms = 16, color='blue')
plt.xlabel('components', fontsize= 10)
plt.ylabel('amount of var', fontsize=10)
# plt.show()

simmovie_1= np.array(intermatgen_1_df.loc['Toy Story (1995)']).reshape(1, -1)
simmovie_2 = np.array(intermatgensecdata.loc['Toy Story (1995)']).reshape(1, -1)
conscore = cosine_similarity(intermatgen_1_df, simmovie_1).reshape(-1)
collabscore = cosine_similarity(intermatgensecdata, simmovie_2).reshape(-1)
hybridsim = ((conscore + collabscore)/2.0)
simdic = {'content': conscore , 'collaborative': collabscore, 'hybrid': hybridsim}
simdf = pd.DataFrame(simdic, index = intermatgen_1_df.index )
simdf.sort_values('content', ascending=False, inplace=True)
#simdf[1:].head(10)

simdf[0:].content.head(10)

simdf[0:].collaborative.head(10)

simdf[0:].hybrid.head(10)

def showprecrec(predrat, k=10, threshold=3.5):
    uet = defaultdict(list)
    for uid, _, true_r, est, _ in predrat:
        uet[uid].append((est, true_r))
    precc = dict()
    recc = dict()
    for uid, urats in uet.items():
        urats.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in urats)
        n_rec_k = sum((est >= threshold) for (est, _) in urats[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in urats[:k])
        precc[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        recc[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
    return precc, recc


reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(filtered_ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=.25)
algo = SVD()
algo.fit(trainset)
predrat = algo.test(testset)
print(accuracy.rmse(predrat, verbose=True))  # rmse
# can calculate better accuracy[precsion and recall bhi calculate] by using K fold shaayd but then hit ratio 83 tha  idk :(((
# time will inccrrease to about 10 min but :)v
# kf = KFold(n_splits=5)
'''for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predrat = algo.test(testset)
    precc, recc = showprecrec(predrat, k=5, threshold=4)
    print('precision and recall at k')
    print(sum(prec for prec in precc.values()) / len(precc))
    print(sum(rec for rec in recc.values()) / len(recc))
    print(accuracy.rmse(predrat, verbose=True))'''

def recommovie(ui):
    if ui in filtered_ratings.userId.unique():
        listforuser = filtered_ratings[filtered_ratings.userId == ui].movieId.tolist()
        d = {k: v for k,v in Mapit.items() if not v in listforuser}
        predlus = []
        for i, j in d.items():
            predic = algo.predict(ui, j)
            predlus.append((i, predic[3]))
        predf = pd.DataFrame(predlus, columns = ['movies', 'ratings'])
        predf.sort_values('ratings', ascending=False, inplace=True)
        predf.set_index('movies', inplace=True)
        return predf.head(10)
    else:
        print("Does not exist")
        return None

recommovie(137072)

uusers = filtered_ratings.userId.unique()

uusers

uusers = uusers[0:50]

uusers
listofusers = uusers.tolist()

listofusers

pickle.dump(algo, open("hybrid.model", 'wb'))

algo2 = pickle.load(open("hybrid.model", 'rb'))

def recommovie2(ui):
    if ui in filtered_ratings.userId.unique():
        listforuser = filtered_ratings[filtered_ratings.userId == ui].movieId.tolist()
        d = {k: v for k,v in Mapit.items() if not v in listforuser}
        predlus = []
        for i, j in d.items():
            predic = algo2.predict(ui, j)
            predlus.append((i, predic[3]))
        predf = pd.DataFrame(predlus, columns = ['movies', 'ratings'])
        predf.sort_values('ratings', ascending=False, inplace=True)
        predf.set_index('movies', inplace=True)
        return predf.head(10)
    else:
        print("Does not exist")
        return None

recommovie2(137072)