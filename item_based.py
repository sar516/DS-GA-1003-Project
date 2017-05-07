import numpy as np
import operator
import copy
import random
import sparse_bag_tools as spt

def adj_cos(dict1, dict2, transpose_bag):
    keys1 = list(dict1.keys())
    keys2 = list(dict2.keys())
    cross_keys = list(set(keys1) & set(keys2))
    all_keys = list(set(keys1) | set(keys2))
    if len(cross_keys) == 0:
        return 0.0
    else:
        means = {key : np.mean(list(transpose_bag[key].values())) for key in cross_keys}
        num = sum([(dict1[key] - means[key])*
                   (dict2[key] - means[key])
                   for key in cross_keys])
        den1 = (sum([(dict1[key] - means[key])**2
                     for key in cross_keys]))**.5
        den2 = (sum([(dict2[key] - means[key])**2
                     for key in cross_keys]))**.5
        try:
            return num/(den1*den2)
        except ZeroDivisionError:
            return 0.0



class item_CF():
    def __init__(self, k = 2, reg = .5):
        self.k = k
        self.reg = reg
        self.fitted = False
    
    
    def dict_NN(self, query):
        dists = [(key, self.get_sim(key, query)) 
                 for key in self.item_bag.keys() if key != query]
        dists.sort(key=lambda pair: pair[1], reverse=True)
        self.neighborhood[query] = [pair[0] for pair in dists]
    
    def fit(self, train_bag):
        self.train_bag = train_bag
        self.item_bag  = spt.sparse_bag_transpose(train_bag)
        self.neighborhood = {}
        self.sims = {}
        item_list = list(self.item_bag.keys())
        for item in self.item_bag.keys():
            i = self.item_bag[item]
            item_list.remove(item)
            self.sims[item] = {k : adj_cos(i, self.item_bag[k], train_bag) for k in item_list}
        self.fitted = True
        return self
    
    
    def get_sim(self, item_1 , item_2):
        if item_1 not in self.sims.keys():
            return 0.0
        elif item_2 not in self.sims.keys():
            return 0.0
        else:
            try:
                return self.sims[item_1][item_2]
            except KeyError:
                return self.sims[item_2][item_1]
    
    def predict(self, test_points, k = None, reg = None):
        
        if not self.fitted:
            raise RuntimeError("You must train filter before predicting data!")
        
        if not k:
            k = self.k
        if not reg:
            reg = self.reg
        
        self.baseline = spt.Baseline(reg).fit(self.train_bag)
        base_predictions  = self.baseline.predict(test_points)
        predictions = {}
        for pair in test_points:
            b_ui = base_predictions[pair]
            user = pair[0]
            item = pair[1]
            if item not in self.neighborhood.keys():
                if item not in self.item_bag.keys():
                    knn = []
                else:
                    self.dict_NN(item)
                    knn = (self.neighborhood[item])
                    
            else:
                if item not in self.item_bag.keys():
                    knn = []
                else:
                    knn = (self.neighborhood[item])
                
                    
            overlap = list(set(knn) & set(list(self.train_bag[user].keys())))
            if len(overlap) >= k:
                N = overlap[:k]
            else:
                N = overlap
                
                
            num = sum([self.get_sim(item, n)*(self.train_bag[user][n]- b_ui) for n in N])
            den = sum([abs(self.get_sim(item, d)) for d in N])
            
            if 0.0 in (num, den):
                predictions[pair] = b_ui
            else:
                predictions[pair] = (num/den) + b_ui
            
        return predictions