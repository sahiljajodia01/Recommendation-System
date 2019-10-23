# A content-based recommendation algorithm.

from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile


def tokenize_string(my_string):
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    movies['tokens'] = [tokenize_string(genre) for genre in movies['genres']]

    return movies


def featurize(movies):
    #creating a vocab of all the unique genres
    vocab = {movie_tokens:idx for idx, movie_tokens in enumerate(sorted(np.unique(np.concatenate(movies.tokens))))}

    # creating df
    df = defaultdict(int)
    for movie_genre in movies.tokens:
        for genre in vocab:
            if genre in movie_genre:
                df[genre]+=1

    all_csr = []
    for idx, movie in enumerate(movies.tokens):
        #print(movie)
        colmn, data, row = [], [], []
        tf = Counter(movie)     # tf
        max_k = tf.most_common(1)[0][1]
        #print(max_k)# max_k
        for genre, freq in tf.items():
            if genre in vocab:
                #row.append(0)
                colmn.append(vocab[genre])
                data.append((freq/max_k)*math.log10(len(movies)/df[genre])) # tf-idf
                X = csr_matrix((np.asarray(data), (np.zeros(shape=(len(data))), np.asarray(colmn))), shape=(1, len(vocab)))

        all_csr.append(X)

    movies['features'] = all_csr

    return movies, vocab


def train_test_split(ratings):
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    a = a.toarray()
    b = b.toarray()
    return (np.dot(a,b.T)) / (np.sqrt(np.sum(np.square(a))) * np.sqrt(np.sum(np.square(b))))



def make_predictions(movies, ratings_train, ratings_test):
    # for every user in Test Set, get the rating from the Train Set
    predictions = []
    for test_userid, test_movieid in zip(ratings_test.userId, ratings_test.movieId):
        # got the test userid & test movieid
        #print("Getting for", test_userid, test_movied)
        weight_ratings = []
        weights = []
        target_user_ratings = []
        for idx, train_user in ratings_train.loc[ratings_train.userId == test_userid, 'movieId': 'rating'].iterrows():
            # got the ratings and movieId for the test userId
            # print(rating_val.movieId, rating_val.rating)
            # print(int(train_user.movieId), int(test_movieid))
            # print(movies.loc[movies.movieId == int(train_user.movieId)].features.values)
            # print(movies.loc[movies.movieId == int(test_movieid)].features.values)

            cos_sim_weight = cosine_sim(movies.loc[movies.movieId == int(train_user.movieId)].features.values[0],
                                        movies.loc[movies.movieId == int(test_movieid)].features.values[0])
            #print(cos_sim_weight)
            weight_ratings.append(train_user.rating * cos_sim_weight)
            weights.append(cos_sim_weight)
            target_user_ratings.append(train_user.rating)


        if np.count_nonzero(weights) > 0:
            #weighted_average = np.sum(weight_ratings)/np.sum(weights)
            predictions.append(np.sum(weight_ratings)/np.sum(weights))
            #print(np.sum(weights))
            #print(weighted_average)
        else:
            predictions.append(ratings_train.loc[ratings_train.userId == test_userid, 'rating'].mean())
            #predictions.append(np.mean(target_user_ratings))

            #print(ratings_train.loc[ratings_train.userId == test_userid, 'rating'].mean())



    return np.asarray(predictions)



def mean_absolute_error(predictions, ratings_test):
    """
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    print("Tokenize\n", movies)
    movies, vocab = featurize(movies)
    print("Featurize\n", movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()