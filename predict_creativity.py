# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:13:39 2021

@author: krist
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

#import features
features = np.loadtxt('novelty.txt')
one_hot_features=np.loadtxt('all_nov.txt')

#import targets from excel sheet as dataframe
#to convert dataframe to numpy, np.asarray(<dataframe>)
targets=np.loadtxt('cm_targets.txt')

train_amt=int(0.8*features.shape[0])
indeces=np.arange(features.shape[0])
np.random.shuffle(indeces)


#no shuffling
# train_feat= one_hot_features[:train_amt,:]
# train_targ= targets[:train_amt,:]

# test_feat= one_hot_features[train_amt:,:]
# test_targ= targets[train_amt:,:]

# reg = LinearRegression().fit(train_feat, train_targ[:,-1:])
# lreg_r_sqaured=reg.score(test_feat, test_targ[:,-1:])
# print("Linear regression R^2 test score: ", reg.score(test_feat, test_targ[:,-1:]))
# print("Linear regression R^2 train score: ", reg.score(train_feat, train_targ[:,-1:]))

# shuffling
one_hot_features_shuf=one_hot_features[indeces]
targets_shuf=targets[indeces]

train_feat= one_hot_features_shuf[:train_amt,:]
train_targ= targets_shuf[:train_amt,:]

test_feat= one_hot_features_shuf[train_amt:,:]
test_targ= targets_shuf[train_amt:,:]

for i in range(targets.shape[1]):
    
    if i==0:
        targ_name= 'Expert 1'
    elif i==1:
        targ_name= 'Expert 2'
    else:
        targ_name= 'Combined'
    
    #GB takes in 1d arrays for y
    test_col=test_targ[:,[i]].flatten()
    train_col=train_targ[:,[i]].flatten()
    
    
    reg = LinearRegression().fit(train_feat, train_col)
    lreg_r_sqaured=reg.score(test_feat, test_col)
    
    test_score= reg.score(test_feat, test_col)
    train_score=reg.score(train_feat, train_col)
    print("{} Linear regression R^2 test score: ".format(targ_name),test_score )
    print("{} Linear regression R^2 train score: ".format(targ_name), train_score )
    
    lr_preds=reg.predict(test_feat)
    
    
    
    plt.plot(test_col,lr_preds,'ro',ms=4)
    plt.ylabel('Predicted Creativity')
    plt.xlabel('Actual Creativity')
    title= '{} Linear Regression Actual v. Predicted Creativity'.format(targ_name)
    plt.title(title)
    plt.savefig(title+'.png')
    plt.show()
    #%% Feature Importance Linear Regression
    # importance = reg.coef_
    # # summarize feature importance
    
    # plt.hist(importance)  # arguments are passed to np.histogram
    # plt.title("Histogram with 'auto' bins")
    # plt.show()
    
    
    #%% Gradient Boosting Regressor
    gb = GradientBoostingRegressor(random_state=0)
    gb.fit(train_feat,train_col)
    
    score=gb.score(test_feat,test_col)
    
    gb.predict(test_feat)
    
    gb_preds= gb.predict(test_feat)
    
    
    print("{} Gradient Boosting Regressor R^2 test score: ".format(targ_name), score)
    
    plt.plot(test_col,gb_preds,'bo',ms=4)
    plt.ylabel('Predicted Creativity')
    plt.xlabel('Actual Creativity')
    title= '{} Gradient Boosting Actual v. Predicted Creativity'.format(targ_name)
    plt.title(title)
    # plt.axis([0,1,0,1])
    # plt.xticks(np.arange(0, 1, 0.1))
    # plt.yticks(np.arange(0, 1,0.1))
    plt.savefig(title+'.png')
    plt.show()
    
    #%% Random Forest Regressor
    
    rf = RandomForestRegressor(random_state=0)
    rf.fit(train_feat,train_col)
    
    score=rf.score(test_feat,test_col)
    rf_preds= rf.predict(test_feat)
    
    print("{} Random Forest Regressor R^2 test score: ".format(targ_name), score)
    
    plt.plot(test_col,rf_preds,'go',ms=4)
    plt.ylabel('Predicted Creativity')
    plt.xlabel('Actual Creativity')
    title= '{} Random Forest Actual v. Predicted Creativity'.format(targ_name)
    plt.title(title)
    # plt.axis([0,1,0,1])
    # plt.xticks(np.arange(0, 1, 0.1))
    # plt.yticks(np.arange(0, 1,0.1))
    plt.savefig(title+'.png')

# #%% Combine GB and RF
# for i in range(targets.shape[1]):
    
#     if i==0:
#         targ_name= 'Scorer 1'
#     elif i==1:
#         targ_name= 'Scorer 2'
#     else:
#         targ_name= 'Combined'
    
#     #GB takes in 1d arrays for y
#     test_col=test_targ[:,[i]].flatten()
#     train_col=train_targ[:,[i]].flatten()
    
#     #plotting
#     plt.plot(test_col,rf_preds,'go', test_col, gb_preds, 'b^', ms=4)
#     plt.ylabel('Predicted Creativity')
#     plt.xlabel('Actual Creativity')
#     plt.title('{} GB and RF Actual v. Predicted Creativity'.format(targ_name))
#     plt.axis([0,1,0,1])
#     plt.xticks(np.arange(0, 1, 0.1))
#     plt.yticks(np.arange(0, 1,0.1))
#     plt.show()
    
# #%% GB for different test train splits

# x=['Scorer 1', 'Scorer 2', 'Combined']
# y=[0.2,0.3,0.4,0.5,0.6,0.7,0.8]

# data=[]
# for i in range(targets.shape[1]):
    
    
#     if i==0:
#         targ_name= 'Scorer 1'
#     elif i==1:
#         targ_name= 'Scorer 2'
#     else:
#         targ_name= 'Combined'
#     row=[]
    
#     for j in range(2,9,1):
        
#         print('Training proportion: ', j/10)
#         print('Testing proportion: ', 1-(j/10))
        
#         train_amt=int(j/10*features.shape[0])
        
#         print("Number of training datapoints: ", train_amt)
#         indeces=np.arange(features.shape[0])
#         np.random.shuffle(indeces)
        
#         # shuffling
#         one_hot_features_shuf=one_hot_features[indeces]
#         targets_shuf=targets[indeces]
        
#         train_feat= one_hot_features_shuf[:train_amt,:]
#         train_targ= targets_shuf[:train_amt,:]
        
#         test_feat= one_hot_features_shuf[train_amt:,:]
#         test_targ= targets_shuf[train_amt:,:]
        
#         #GB takes in 1d arrays for y
#         test_col=test_targ[:,[i]].flatten()
#         train_col=train_targ[:,[i]].flatten()
        
#         gb = GradientBoostingRegressor(random_state=0)
#         gb.fit(train_feat,train_col)
        
#         score=gb.score(test_feat,test_col)
        
#         gb.predict(test_feat)
        
#         gb_preds= gb.predict(test_feat)
        
#         row.append(score)
        
        
#         print("{} Gradient Boosting Regressor R^2 test score, Train: {}: ".format(targ_name, j/10), score)
        
#         # plt.plot(test_col,gb_preds,'bo',ms=4)
#         # plt.ylabel('Predicted Creativity')
#         # plt.xlabel('Actual Creativity')
#         # plt.title('{} Gradient Boosting Actual v. Predicted Creativity, Train {}'.format(targ_name, j/10))
#         # plt.axis([0,1,0,1])
#         # plt.xticks(np.arange(0, 1, 0.1))
#         # plt.yticks(np.arange(0, 1,0.1))
#         # plt.show()
#     data.append(row)

# results = pd.DataFrame(data, index=x, columns=y)
 
# results.to_excel('gradient_boosting_test_train.xlsx')                 
                  