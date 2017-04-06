import numpy as np
import pandas as pd
from collections import Counter
import operator
import copy
import random

def bag_of_games(data_set):
    user_ids = data_set["user_id"].unique()
    user_bag = {}#.fromkeys(map((lambda x: str(x)),user_ids), {})
    
    
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
    
    
    

def baseline(train_bag, query_bag, heldout, user_reg=25, game_reg=25):
    total_bag = copy.deepcopy(train_bag)
    total_bag.update(query_bag)
    n = 0
    total = 0
    for userid in train_bag:
        gamelist = train_bag[userid]
        n += len(list(gamelist.keys()))
        total += sum(gamelist.values())
    else:
        total_avg = total/n
        
    user_baseline = {}
    for userid in total_bag:
        user = total_bag[userid]
        user_baseline[userid] = (sum(user.values()) - len(user)*total_avg)/(len(user)+user_reg)
    
    predictions = {}
    for test_point in heldout.keys():
        test_user = test_point[0]
        test_game = test_point[1]
        bu = user_baseline[test_user]
        n = 0
        bi = 0
        for userid in train_bag:
            user = train_bag[userid]
            if test_game in user.keys():
                bi += user[test_game] - user_baseline[userid] - total_avg
                n += 1
        else:
            bi = bi/(n + game_reg)
        
        
        predictions[test_point] = total_avg + bu + bi
    return predictions


def get_metrics(actuals, preds):
    abs_diff = {}
    square_diff = {}
    for point in preds.keys():
        diff = float(actuals[point]) - float(preds[point])
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
            if game not in transpose:
                transpose[game] = {user : rating}
            else:    
                transpose[game].update({user : rating})
    return transpose