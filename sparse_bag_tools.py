import numpy as np
import pandas as pd
from collections import Counter
import operator
import copy
import random

def bag_of_games(data_set):
    user_ids = data_set["user_id"].unique()
    user_bag = {}
    
    
    for user in user_ids:
        
        user_bag[str(user)] = {}
        for row in data_set[data_set["user_id"] == user][["game", "value"]].itertuples():
            user_bag[str(user)][row[1]] = row[2]
            
    return user_bag


def leave_n_in(bag, n=1, p = .25):
    copy_bag = copy.deepcopy(bag)
    user_ids = list(copy_bag.keys())
    
    N = 0
    targets = []
    for user in bag.keys():
        game_list = bag[user] 
        j = len(game_list)
        N += j
        if j > n:
            targets.append(user)
            
                
    k = int(p*N)
    popped = []
    removed = {}
    i = 0
    h = len(targets)
    while i != k:
        rand_1 = int(h*(random.random()))
        user = targets[rand_1]
        gamelist = copy_bag[user]
        j = len(gamelist) 
        if j <= n:
            pass
        else:
            if user not in popped:
                popped.append(user)
            rand_2 = int(j*random.random())
            games  = list(gamelist.keys())
            selected = games[rand_2]
            removed[(user, selected)] = gamelist.pop(selected)   
            i += 1
    else:
        query_bag = {}
        for user in popped:
            query_bag[user] = copy_bag.pop(user)
    return copy_bag, removed, query_bag
    

def mean_norm(bag, n = 1, p = .25):
    train_bag, test_points, query_bag = leave_n_in(bag, n, p)
    mean = 0
    N = 0
    for user in train_bag.keys():
        games_list = train_bag[user]
        mean += sum(games_list.values())
        N += len(games_list)
    else:
        mean = mean/N
        
    for user in train_bag.keys():
        games_list = train_bag[user]
        for game in games_list.keys():
            games_list[game] = games_list[game]/mean
    
    for pair in test_points.keys():
        test_points[pair] = test_points[pair]/mean
        
    return train_bag, test_points, query_bag 
    

class Baseline():
    
    def __init__(self, reg = 25):
        self.reg = reg
        self.fitted = False
        
    def get_bu(self, users_list):
        bu = (sum([rating  - self.mu  for rating in users_list.values()])/
              (len(users_list)+self.reg))
        return bu
    
    def get_bi(self, items_list):
        bi = (sum([rating - self.user_baselines[user] - self.mu
                   for user, rating in items_list.items()])/(len(items_list)+self.reg))
        return bi
    
    
    def fit(self, train_bag):
        self.train_bag = train_bag
        self.item_bag  = sparse_bag_transpose(train_bag)
        self.mu = (sum([sum(v.values()) for v in self.train_bag.values()])/
                   sum([len(v.keys()) for v in self.train_bag.values()]))
        self.user_baselines = {user : self.get_bu(users_list) 
                               for user, users_list in self.train_bag.items()}
        self.item_baselines = {item : self.get_bi(items_list)
                               for item, items_list in self.item_bag.items()}
        
        self.fitted = True
        
        return self
    
    
    def predict(self, test_points, query_bag):
        predictions = {}
        query_baselines = {user : self.get_bu(users_list) 
                           for user, users_list in query_bag.items()}
        for pair in test_points:
            user, item = pair
            b_u = query_baselines[user]
            b_i = self.item_baselines.get(item, 0)
            predictions[pair] = self.mu + b_u + b_i
                
        return predictions

def get_metrics(actuals, preds):
    abs_diff = {}
    square_diff = {}
    for point in preds.keys():
        diff = actuals[point] - preds[point]
        abs_diff[point] = abs(diff)
        square_diff[point] = diff*diff
    MAE = np.array(list(abs_diff.values())).mean()
    MSE = np.array(list(square_diff.values())).mean()
    RMSE = MSE**.5
    return MAE, RMSE

def sparse_bag_transpose(bag):
    transpose = {}
    for user, games_list in bag.items():
        for game, rating in games_list.items():
            if game not in transpose.keys():
                transpose[game] = {user : rating}
            else:    
                transpose[game].update({user : rating})
    return transpose

def dict_dist(dict1, dict2):
    keys1 = list(dict1.keys())
    keys2 = list(dict2.keys())
    all_keys = list(set(keys1) | set(keys2))
    dist = 0
    for key in all_keys:
        dist += (dict1.get(key,0)-dict2.get(key,0))**2
        
    return dist**.5


def knn_dict(neighbors, query, k = 2):
    keys = list(neighbors.keys())
    dists = np.zeros(len(neighbors))
    knn = {}
    if k > len(keys):
        raise ValueError("k greater than number of training items")
        
    for i in range(len(keys)):
        
        key = keys[i]
        dists[i] = dict_dist(neighbors[key], query)
        
    for j in range(k):
        
        m = np.argmin(dists)
        dists[m] = 10**10
        selected = keys[m] 
        knn[selected] = neighbors[selected]
        
    return knn

def dict_dot(dict1, dict2):
    keys1 = list(dict1.keys())
    keys2 = list(dict2.keys())
    cross_keys = list(set(keys1) & set(keys2))
    prod = 0
    if len(cross_keys) == 0:
        return prod
    else:
        for key in cross_keys:
            prod += dict1[key]*dict2[key]
        
        return prod
    
    
    
def cos_sim(dict1, dict2):
    try:
        return dict_dot(dict1, dict2)/(dict_dist(dict1,{})*dict_dist(dict2,{}))
    except ZeroDivisionError:
        return 0