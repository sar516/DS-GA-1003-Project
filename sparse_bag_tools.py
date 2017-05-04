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

def preprocessing(bag):
    user_count = {user : len(items) for user, items in bag.items()}
    k ={}
    for key, value in user_count.items():
        if value > 1:
            k[key] = value
    exp_bag = {user: bag[user] for user in k.keys()}
    return exp_bag


def leave_n_in(bag, n=1, p = .25):
    copy_bag = copy.deepcopy(bag)
    
    N = sum([len(lists) for lists in bag.values()])
    item_bag = sparse_bag_transpose(bag)
    item_count = {item : len(users) for item , users in item_bag.items()}
    k = int(p*N)
    popped = []
    removed = {}
    i = 0
    list_of_user = list(copy_bag.keys())
    black_list = []
    
    while i != k:
        h = len(list_of_user)
        will_remove_item = False
        rand = int(h*(random.random()))
        user = list_of_user[rand]
        
            
        items = copy_bag[user]
        j = len(items) 
        
        if j <= n:
            list_of_user.remove(user)
        else:
            item_list = list(items.keys())
            rand_2 = int(j*(random.random()))
            item = item_list[rand_2]
            if item_count[item] > 1:
                removed[(user, item)] = items.pop(item)
                j -= 1
                item_count[item] -= 1
                if j == n:
                    list_of_user.remove(user)
                
                i += 1
            
    else:
        return copy_bag, removed

def norm(bag, func = None):
    copy_bag = copy.deepcopy(bag)
    if not func:
        func = lambda x : np.log10(1 + x*60)
        
    for user in copy_bag.keys():
        games_list = copy_bag[user]
        copy_bag[user] = {k: func(v) for k, v in games_list.items()}
        
    return copy_bag
    

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
    
    
    def predict(self, test_points):
        predictions = {}
        for pair in test_points:
            user, item = pair
            b_u = self.user_baselines[user]
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
        dists[i] = cos_sim(neighbors[key], query)
        
    for j in range(k):
        
        m = np.argmax(dists)
        dists[m] = -10**10
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