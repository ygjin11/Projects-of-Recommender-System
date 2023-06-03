import math
import random
import pandas as pd
from collections import defaultdict
from operator import itemgetter
from tool import LoadMovieLensData, PreProcessData
from model import ItemCF

if __name__ == "__main__":

    # load data
    train, test = LoadMovieLensData("../Dataset/ml-1m/ratings.dat", 0.8)
    print("train data size: %d, test data size: %d" % (len(train), len(test)))

    # train
    ItemCF = ItemCF(train, similarity='cosine', norm=True)
    ItemCF.train()

    # inference
    print(ItemCF.recommend(1, 5, 80))
    print(ItemCF.recommend(2, 5, 80))
    print(ItemCF.recommend(3, 5, 80))
    print(ItemCF.recommend(4, 5, 80))

