"""
This module holds the class necessary to make the Matrix Factorization models.
Also the structure of this code was heavily inspired by:
https://gist.github.com/EthanRosenthal/a293bfe8bbe40d5d0995
"""

import numpy as np
import operator
import copy
import random
import sparse_bag_tools as spt


class matrix_factor():
    """
    This class is used to build the matrix factors using either Alternating Least Squares or 
    Stochastic Gradient Descent
    """
    def __init__(self, k = 2, method = "als"):
        """
        Here we iniailize the class with the number of latent factors k and
        which method to use
        """
        self.k = k
        self.fitted = False
        if method not in ["sgd", "als"]:
            raise RuntimeError("%s is not a valid method for this class" % (method))
        else:
            self.method = method
            
            
    def fit(self, train_bag, steps = 10, beta = 1, alpha = 0.01):
        """
        This begains training the data to the method specified before
        where beta is the regularization term and alpha is the step size for 
        """
        
        # This block of code is used to iniatilize the factor matrices and setup the dictionaries that 
        # will allow us to jump from the nested dictionary representation to a numpy array
        self.alpha = alpha
        self.item_bag  = spt.sparse_bag_transpose(train_bag)
        self.user_list  = list(train_bag.keys())
        self.item_list = list(self.item_bag.keys())
        self.W = np.zeros((len(self.user_list), len(self.item_list)))
        self.R = np.zeros((len(self.user_list), len(self.item_list)))
        self.U = np.random.rand(len(self.user_list), self.k)
        self.I = np.random.rand(self.k, len(self.item_list))
        self.U_index = {self.user_list[u]: u for u in range(len(self.user_list))}
        self.I_index = {self.item_list[i]: i for i in range(len(self.item_list))}
        self.indices = []
        self.beta = beta
        
        
            
        # Here we translate the rating matrix from the nested dictionary to a numpy array 
        for user in train_bag.keys():
            u = self.U_index[user]
            for item in train_bag[user].keys():
                i = self.I_index[item]
                self.R[u, i] = train_bag[user][item]
                self.W[u, i] = 1
                self.indices.append((u, i))
                
        self.Beta = self.beta*np.eye(self.k)
        self.mean = np.sum(self.R)/np.sum(self.W)
        self.errs = []
        
        
        # Here the steps are implemented
        for step in range(steps):
            
            if self.method == "als":
                self.als_step()
            elif self.method == "sgd":
                np.random.shuffle(self.indices)
                self.sgd_step()

            
            
            if self.errs[-1] < 0.001:
                break
                
        return self
    
    
    def predict_point(self, u, i):
        """
        This method predicts the rating for a user-item pair
        """
        prediction = np.dot(self.U[u,:], self.I[:,i])
        return prediction
    
    
    def als_step(self):
        """
        This method implements a single als step 
        """
        inv_for_U = np.linalg.inv(np.dot(self.I, self.I.T) + self.Beta)
        for u in range(len(self.user_list)):
            self.U[u, :] = np.dot(np.dot(self.R[u,:].reshape(1,-1), self.I.T),
                                  inv_for_U).reshape(self.k)
            
            
                
        inv_for_I = np.linalg.inv(np.dot(self.U.T, self.U) + self.Beta)
        for i in range(len(self.item_list)):
            self.I[:,i] = np.dot(np.dot(self.R[:, i].reshape(1,-1), self.U),
                                 inv_for_I).reshape(self.k)
            
            
            
        err = (np.sum((self.R - (self.W*np.dot(self.U,self.I)))**2)/(np.sum(self.W)))**.5
        self.errs.append(err)
        
        
    def sgd_step(self):
        """
        This method implements a single sgd step 
        """
        for index in self.indices:
            u, i = index
            prediction = self.predict_point(u, i)
            e = self.R[u, i] - prediction
            
            
            self.U[u, :] = self.U[u, :] + self.alpha*(e*self.I[:, i] - self.beta*self.U[u, :])
            self.I[:, i] = self.I[:, i] + self.alpha*(e*self.U[u, :] - self.beta*self.I[:, i])
            
        err = np.sum((self.R - (self.W*np.dot(self.U,self.I)))**2)/(np.sum(self.W))
        self.errs.append(err)
        
    
    def fold_in_user(self, users_list, user_id):
        """
        This function folds in a new user into the user latent factor matrix
        """
        self.U_index[user_id] = len(self.user_list)
        self.user_list.append(user_id)
        ru = np.zeros(len(self.item_list))
        for item, rating in users_list.items():
            ru[self.I_index[item]] = rating
            
        u = np.dot(self.I, ru).reshape(1, -1)
        self.U = np.append(self.U, u, axis = 0)
        
    def predict(self, test_points, query_bag = {}):
        """
        This method makes predictions for user-item pairs in test_points 
        """
        predictions = {}
        for pair in test_points:
            user = pair[0]
            item = pair[1]
            u, i = self.U_index[user], self.I_index[item]
            predictions[pair] = self.predict_point(u, i)
        return predictions