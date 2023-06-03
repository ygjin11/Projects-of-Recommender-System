import math
import random
import pandas as pd
from collections import defaultdict
from operator import itemgetter

class UserCF(object):
    """ Item based Collaborative Filtering Algorithm Implementation"""
    def __init__(self, trainData, similarity="cosine", norm=True):
        self._trainData = trainData
        self._similarity = similarity
        self._isNorm = norm
        self._uesrSimMatrix = dict() 

    def similarity(self):
        N = defaultdict(int) # num of items liked by user
        for user, items in self._trainData.items():
            N[user] = len(items)

        for user, items_user in self._trainData.items():
            self._uesrSimMatrix.setdefault(user, dict())
            for otheruser, items_otheruser in self._trainData.items():
                self._uesrSimMatrix.setdefault(otheruser, dict())
                if user == otheruser:
                    continue
                interseclen = len(items_user & items_otheruser)
                # find whether items liked user & otheruser exists
                self._uesrSimMatrix[user].setdefault(otheruser, interseclen) 
                self._uesrSimMatrix[otheruser].setdefault(user, interseclen) 
        
        for i, related_users in self._uesrSimMatrix.items():
            # i: a user
            # related_users: users has same users as i 
            for j, cij in related_users.items():
                # j of related_users
                # cij: nums of items shared
                self._uesrSimMatrix[i][j] = cij / math.sqrt(N[i]*N[j]) # calculate similarity
        # norm itemSimMatrix
        if self._isNorm:
            for i, relations in self._uesrSimMatrix.items():
                max_num = relations[max(relations, key=relations.get)]
                self._uesrSimMatrix[i] = {k : v/max_num for k, v in relations.items()}  

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
        related_usersandsim = self._uesrSimMatrix[user]
        items = self._trainData[user]
        for ruser, sim in sorted(self._uesrSimMatrix[user].items(), key=itemgetter(1), reverse=True)[:K]:
            items_ruser = self._trainData[ruser]
            for i in items_ruser:
                if i in items:
                    continue 
                recommends.setdefault(i, 0.)
                recommends[i] += sim
        # return items recommended
        return dict(sorted(recommends.items(), key=itemgetter(1), reverse=True)[:N])

