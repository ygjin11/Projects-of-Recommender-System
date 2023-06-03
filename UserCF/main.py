import math
import random
import pandas as pd
from collections import defaultdict
from operator import itemgetter
from tool import LoadMovieLensData, PreProcessData
from model import UserCF

if __name__ == "__main__":

    # load data
    train, test = LoadMovieLensData("../Dataset/ml-1m/ratings.dat", 0.8)
    print("train data size: %d, test data size: %d" % (len(train), len(test)))

    # train
    UserCF = UserCF(train, similarity='cosine', norm=True)
    UserCF.train()

    # inference
    print(UserCF.recommend(1, 5, 80))
    print(UserCF.recommend(2, 5, 80))
    print(UserCF.recommend(3, 5, 80))
    print(UserCF.recommend(4, 5, 80))

