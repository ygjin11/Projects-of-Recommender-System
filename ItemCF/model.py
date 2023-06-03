import math
import random
import pandas as pd
from collections import defaultdict
from operator import itemgetter

class ItemCF(object):
    """ Item based Collaborative Filtering Algorithm Implementation"""
    def __init__(self, trainData, similarity="cosine", norm=True):
        self._trainData = trainData
        self._similarity = similarity
        self._isNorm = norm
        self._itemSimMatrix = dict() 

    def similarity(self):
        N = defaultdict(int) # num of users liking this item
        for user, items in self._trainData.items():
            for i in items:
                self._itemSimMatrix.setdefault(i, dict()) 
                N[i] += 1 
                for j in items:
                    if i == j:
                        continue
                    self._itemSimMatrix[i].setdefault(j, 0) # find whether users like item i & j exists
                    if self._similarity == "cosine":
                        self._itemSimMatrix[i][j] += 1  # num of users like item i & j + 1 
                    elif self._similarity == "iuf": # similar
                        self._itemSimMatrix[i][j] += 1. / math.log1p(len(items) * 1.)
        for i, related_items in self._itemSimMatrix.items():
            # i: a item
            # related_items: items has same users as i 
            for j, cij in related_items.items():
                # j of related_items
                # cij: nums of users shared
                self._itemSimMatrix[i][j] = cij / math.sqrt(N[i]*N[j]) # calculate similarity
        # print(self._itemSimMatrix[1][2])
        # morm itemSimMatrix
        if self._isNorm:
            for i, relations in self._itemSimMatrix.items():
                max_num = relations[max(relations, key=relations.get)]
                self._itemSimMatrix[i] = {k : v/max_num for k, v in relations.items()}  
        # print(self._itemSimMatrix[1][2])    

    def train(self):
        self.similarity()

    def recommend(self, user, N, K):
        """
        :param user: user to be recommended
        :param N: numbers of items
        :param K: number of items that is similar to each item user likes
        :return: items predicted by itemcf model
        """
        recommends = dict()
        # items that user likes 
        items = self._trainData[user]
        for item in items:
            # find some items similar to 'item' 
            for i, sim in sorted(self._itemSimMatrix[item].items(), key=itemgetter(1), reverse=True)[:K]:
                if i in items:
                    continue 
                recommends.setdefault(i, 0.)
                recommends[i] += sim
        # return items recommended
        return dict(sorted(recommends.items(), key=itemgetter(1), reverse=True)[:N])

