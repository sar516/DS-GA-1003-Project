import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import os

from sparse_bag_tools import *

raw_games = pd.read_csv("steam-200k.csv", names =["user_id", "game", "behavior", "value", "misc."])
playtime = raw_games[raw_games["behavior"] == "play" ]
purchased = raw_games[raw_games["behavior"] == "purchase" ]

playtime_bag = norm(preprocessing(bag_of_games(playtime)))

train_bag, test_points = leave_n_in(playtime_bag)


def mean_rating(dict):
    count = len(dict)
    sum_rating = sum([i for i in dict.values()])
    return (sum_rating * 1.0) / count

class user_based_CF():
    '''
    User based CF recommendation system class
    :method fit: fit model
    '''

    def __init__(self, train_bag):
        self.sim_mat = {}
        self.train_bag = train_bag
        self.fitted = False
        self.user_dict = {}
        i = 0
        for user in list(self.train_bag.keys()):  # Mapping users to index of sim matrix
            self.user_dict[user] = i
            i += 1

    def knn1(self, query):

        
        
        
        # find the valid neighbors
        """neighbors = {ref_user : train_bag[ref_user] for}
        for ref_user in list(train_bag.keys()):
            for ref_item in list(train_bag[ref_user].keys()):
                if (ref_item == item) and (ref_user != user):
                    neighbors[ref_user] = train_bag[ref_user]

        valid_nb = list(neighbors.keys())
        knn = {}
        if k > len(valid_nb):
            k = len(valid_nb)

        dists = np.zeros(len(valid_nb))
        for i in range(len(valid_nb)):
            key = valid_nb[i]
            dists[i] = self.sim_mat[self.user_dict[key], self.user_dict[user]]

        for j in range(k):
            m = np.argmax(dists)
            dists[m] = -10 ** 10
            selected = valid_nb[m]
            knn[selected] = neighbors[selected]

        return knn"""
        
        dists = [(key, self.get_sim(key, query)) 
                 for key in self.train_bag.keys() if key != query]
        dists.sort(key=lambda pair: pair[1], reverse=True)
        self.neighborhood[query] = [pair[0] for pair in dists]

    # def knn2(self, query, k=2):
    #     keys = list(neighbors.keys())
    #     dists = np.zeros(len(neighbors))
    #     knn = {}
    #     if k > len(keys):
    #         raise ValueError("k greater than number of training items")
    #
    #     for i in range(len(keys)):
    #         key = keys[i]
    #         dists[i] = cos_sim(neighbors[key], query)
    #
    #     for j in range(k):
    #         m = np.argmax(dists)
    #         dists[m] = -10 ** 10
    #         selected = keys[m]
    #         knn[selected] = neighbors[selected]
    #
    #     return knn

    def fit(self):
        print('Fitting Start!!!\n')
        self.item_bag  = sparse_bag_transpose(train_bag)
        count = 0
        n = len(self.train_bag.keys())
        self.sim_mat = np.zeros((n, n))
        n = n**2
        for user1 in list(self.train_bag.keys()):
            for user2 in list(self.train_bag.keys()):
                self.sim_mat[self.user_dict[user1], self.user_dict[user2]] = cos_sim(self.train_bag[user1], self.train_bag[user2])
                count += 1
                if count in [n/10,n/5,n/3,n/2,int(n/1.67),int(n/1.43),int(n/1.25),int(n/1.11)]:
                    print('Fitting Progress:' + str(count*1.0/n) + '\n')

        self.fitted = True
        self.neighborhood = {}
        print('Fitting Complete!!!\n')
        return self
    
    
    def get_sim(self, user1, user2):
        return self.sim_mat[self.user_dict[user1], self.user_dict[user2]]
    
    
    def predict(self, test_points, k=2):
        predictions = {}
        for pair in test_points.keys():
            user, item = pair
            pred_bag = self.train_bag[user] #extact the target user's data
            pred = 0  # upper of the mean weighted average equation
            abs_sim = 0 # lower of the mean weighted average equation

            if user not in self.neighborhood.keys():
                if user not in self.train_bag.keys():
                    knn = []
                else:
                    self.knn1(user)
                    knn = (self.neighborhood[user])
                    
            else:
                if user not in self.train_bag.keys():
                    knn = []
                else:
                    knn = (self.neighborhood[user])
              
            
            
            overlap = list(set(knn) & set(list(self.item_bag[item].keys())))
            if len(overlap) >= k:
                N = overlap[:k]
            else:
                N = overlap
                
                
            pred = sum([self.get_sim(user, u)*(self.item_bag[item][u]- mean_rating(self.train_bag[u])) 
                        for u in N])
            abs_sim = sum([abs(self.get_sim(user, u)) for u in N])
            
            if abs_sim == 0:
                predictions[pair] = mean_rating(pred_bag)
                continue
            predictions[pair] = mean_rating(pred_bag) + (pred/abs_sim)

        return predictions

    def predict2(self, test_points, k=2):
        predictions = {}
        for pair in test_points.keys():
            user, item = pair
            pred_bag = self.train_bag[user]  # extact the target user's data
            pred = 0  # upper of the mean weighted average equation
            abs_sim = 0  # lower of the mean weighted average equation

            if user not in self.neighborhood.keys():
                if user not in self.train_bag.keys():
                    knn = []
                else:
                    self.knn1(user)
                    knn = (self.neighborhood[user])

            else:
                if user not in self.train_bag.keys():
                    knn = []
                else:
                    knn = (self.neighborhood[user])

            overlap = list(set(knn))
            if len(overlap) >= k:
                N = overlap[:k]
            else:
                N = overlap

            for u in N:
                if u in self.item_bag[item]:
                    pred += self.get_sim(user, u) * (self.item_bag[item][u] - mean_rating(self.train_bag[u]))
                    abs_sim += abs(self.get_sim(user, u))
                else:
                    pred += self.get_sim(user, u) * (0 - mean_rating(self.train_bag[u]))
                    abs_sim += abs(self.get_sim(user, u))
            # pred = sum([self.get_sim(user, u) * (self.item_bag[item][u] - mean_rating(self.train_bag[u]))
            #             for u in N])
            # abs_sim = sum([abs(self.get_sim(user, u)) for u in N])

            if abs_sim == 0:
                predictions[pair] = mean_rating(pred_bag)
                continue
            predictions[pair] = mean_rating(pred_bag) + (pred / abs_sim)

        return predictions

if __name__ == "__main__":
    ub = user_based_CF(train_bag)
    ub.fit()

    #path = "../data.dat"
    # np.savetxt(os.path.abspath(path), ub.sim_mat)
    #ub.sim_mat = np.loadtxt(os.path.abspath(path))

    MAE = []
    MAE2 = []
    RMSE = []
    RMSE2 = []
    for k in range(2,20):
        pred = ub.predict(test_points, k)
        pred2 = ub.predict2(test_points, k)
        mae, rmse = get_metrics(test_points, pred)
        mae2, rmse2 = get_metrics(test_points, pred2)
        MAE.append(mae)
        MAE2.append(mae2)
        RMSE.append(rmse)
        RMSE2.append(rmse2)
        print('k = ' + str(k) + ': MAE = ' + str(mae) + ' MAE2 = ' + str(mae2) + ' RMSE = ' + str(rmse)) + 'RMSE2 = ' + str(rmse2)
        print('\n')


    plt.xlabel('K')
    plt.ylabel('Error')
    plt.title('MAE and RMSE for two methods')
    plt.plot(range(2,20), MAE, 'r', label = "MAE for method 2")
    plt.plot(range(2,20), MAE2, 'b', label = "MAE for method 1")
    plt.plot(range(2,20), RMSE, 'g', label = "RMSE for method 2")
    plt.plot(range(2,20), RMSE2, 'k', label = "RMSE for method 1")
    plt.legend()
    plt.savefig("user_twomethods.jpg")
    plt.show()

    # plt.plot(range(2,20), RMSE)
    # plt.xlabel('k')
    # plt.ylabel('RMSE')
    # plt.show()
