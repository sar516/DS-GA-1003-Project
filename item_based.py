"""
This module holds a class and the function necessary to make the Item-based Collaborative Filter
"""

import numpy as np
import operator
import copy
import random
import sparse_bag_tools as spt

def adj_cos(dict1, dict2, transpose_bag):
    """
    This function computes the adjusted similarity between two dictionary vectors
    """
    
    # Here we leverage the fact that we only need keys the that dictionary vectors
    # have in common to compute the similarity 
    keys1 = list(dict1.keys())
    keys2 = list(dict2.keys())
    cross_keys = list(set(keys1) & set(keys2))
    all_keys = list(set(keys1) | set(keys2))
    if len(cross_keys) == 0:
        return 0.0
    else:
        
        means = {key : np.mean(list(transpose_bag[key].values())) for key in cross_keys}
        
        # Compute the numerator of the adjusted cosine
        num = sum([(dict1[key] - means[key])*
                   (dict2[key] - means[key])
                   for key in cross_keys])
        
        # Compute the two terms that need to be multiplied together to calculate
        # the denominator of the adjusted cosine
        den1 = (sum([(dict1[key] - means[key])**2
                     for key in cross_keys]))**.5
        den2 = (sum([(dict2[key] - means[key])**2
                     for key in cross_keys]))**.5
        try:
            return num/(den1*den2)
        except ZeroDivisionError:
            return 0.0



class item_CF():
    """
    This class builds a item base collaborative filter
    """
    
    def __init__(self, k = 2, reg = .5):
        self.k = k
        self.reg = reg
        self.fitted = False
    
    
    def dict_NN(self, query):
        """
        This method finds the items that have been rated by the same users
        and then orders them by similarity
        """
        dists = [(key, self.get_sim(key, query)) 
                 for key in self.item_bag.keys() if key != query]
        dists.sort(key=lambda pair: pair[1], reverse=True)
        self.neighborhood[query] = [pair[0] for pair in dists]
    
    def fit(self, train_bag):
        """
        This method fit the training data to the model by buliding the item-item 
        """
        self.train_bag = train_bag
        self.item_bag  = spt.sparse_bag_transpose(train_bag)
        self.neighborhood = {}
        self.sims = {}
        item_list = list(self.item_bag.keys())
        
        # In this for loop we build the item-item sim matrix and store them in
        # a triangluar matrix to save time
        for item in self.item_bag.keys():
            i = self.item_bag[item]
            item_list.remove(item)
            self.sims[item] = {k : adj_cos(i, self.item_bag[k], train_bag) for k in item_list}
        self.fitted = True
        return self
    
    
    def get_sim(self, item_1 , item_2):
        """
        This method retrieves similarities from the triangluar matrix built during fitting
        """
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
        """
        This method makes predictions for user-item pairs in test_points
        We also added the option of changing k and the regulrization term reg 
        without having to retrain the model
        """
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
            
            # Here we either retrieve or make the list of neighbors need to make the prediction
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
                
            # Here we compute the the numerator and denominator for the prediction equation    
            num = sum([self.get_sim(item, n)*(self.train_bag[user][n]- b_ui) for n in N])
            den = sum([abs(self.get_sim(item, d)) for d in N])
            
            if 0.0 in (num, den):
                predictions[pair] = b_ui
            else:
                predictions[pair] = (num/den) + b_ui
            
        return predictions