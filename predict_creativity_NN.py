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
from datetime import date
import tensorflow as tf
from tensorflow import keras as k
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras import Sequential
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import math


average=True
maximum= False
minimum=False
plot=False
to_excel=False
pca_red=True
svs_and_text=False
just_NN=False
NN=False
just_text=True

#default folder
folder='individual vals'


#import features
features = 'all_nov'

svs_features=np.loadtxt('{}.txt'.format(features))

# text_features=np.loadtxt('text_embeddings.txt')
text_features=pd.read_excel('full_text_embeddings.xlsx',engine='openpyxl')

text_features.drop('Unnamed: 0', axis=1)
#import targets from excel sheet as dataframe
#to convert dataframe to numpy, np.asarray(<dataframe>)

targets=pd.read_excel('Edited_Creativity_Metrics.xlsx', sheet_name='Sheet2', engine='openpyxl')
# Pandas uses iloc to locate based on index number rather than label (loc)
# Targets are a datagrame and features are just an array
#%% PCA Dimensionality reduction
if pca_red: 
    pca=PCA(n_components=15)
     
    pca.fit(svs_features)
    
    new_svs_feat=pca.fit_transform(svs_features)
    
    # LOOK OVER THIS LATER 
    one_hot_features=new_svs_feat
    text_pca=PCA(n_components=30)
    text_pca.fit(text_features)
    
    new_text_feat=text_pca.fit_transform(text_features)
else:
    one_hot_features=svs_features

if svs_and_text:
    one_hot_features=np.append(one_hot_features, new_text_feat, axis=1)

if just_text:
    one_hot_features=new_text_feat
#%% If average

if average:
    folder='average vals'
    targets=pd.read_excel('Edited_Creativity_Metrics.xlsx', sheet_name='Sheet4', engine='openpyxl')

#%% If max

if maximum:
    folder='maximum vals'
    data=[]
    for i in range(0,10,2):
        a=targets.iloc[:,i]
        b=targets.iloc[:,i+1]
        c=np.maximum(a,b)
        data.append(c)
    targets_array=np.array(data).T
    targets=pd.DataFrame(data=targets_array)
        
#%% If min

if minimum:
    folder='minimum vals'
    data=[]
    for i in range(0,10,2):
        a=targets.iloc[:,i]
        b=targets.iloc[:,i+1]
        c=np.minimum(a,b)
        data.append(c)
    targets_array=np.array(data).T
    targets=pd.DataFrame(data=targets_array)
          

#%% Remove NaNs
# Row 753

targets=targets.drop(753, axis=0)
# one_hot_features=np.delete(one_hot_features, (753), axis=0)

#%%
train_amt=int(0.8*one_hot_features.shape[0])
indeces=np.arange(one_hot_features.shape[0])
np.random.shuffle(indeces)


#no shuffling
train_feat= one_hot_features[:train_amt,:]
train_targ= targets.iloc[:train_amt,:]

test_feat= one_hot_features[train_amt:,:]
test_targ= targets.iloc[train_amt:,:]


# shuffling
# one_hot_features_shuf=one_hot_features[indeces]
# targets_shuf=targets.iloc[indeces]

# train_feat= one_hot_features_shuf[:train_amt,:]
# train_targ= targets_shuf.iloc[:train_amt,:]

# test_feat= one_hot_features_shuf[train_amt:,:]
# test_targ= targets_shuf.iloc[train_amt:,:]

#%% Neural Net
if NN:
    num_inputs=one_hot_features.shape[1]
    
    model = Sequential()
    model.add(k.layers.BatchNormalization())
    
    model.add(Dense(128, input_dim=num_inputs, activation='relu'))
    model.add(k.layers.BatchNormalization())
    # model.add(k.layers.Activation(tf.keras.activations.selu))
    
    model.add(Dense(256, activation='relu'))
    model.add(k.layers.BatchNormalization())
    #model.add(k.layers.Activation(tf.keras.activations.selu))
    
    model.add(Dense(512, activation='relu'))
    model.add(k.layers.BatchNormalization())
    #model.add(k.layers.Activation(tf.keras.activations.selu))
    
    model.add(Dense(1024, activation='relu'))
    model.add(k.layers.BatchNormalization())
    #model.add(k.layers.Activation(tf.keras.activations.selu))
    
    
    model.add(Dense(512, activation='relu'))
    model.add(k.layers.BatchNormalization())
    #model.add(k.layers.Activation(tf.keras.activations.selu))
    
    model.add(Dense(256, activation='relu'))
    model.add(k.layers.BatchNormalization())
    #model.add(k.layers.Activation(tf.keras.activations.selu))
    
    model.add(Dense(128, activation='relu'))
    model.add(k.layers.BatchNormalization())
    #model.add(k.layers.Activation(tf.keras.activations.selu))
    
    model.add(Dense(1, activation = 'relu'))
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

if just_NN:
    NN_data=[]
    for column in test_targ:
    
        targ_name= column
    
        # Target value from specific column that we'll use for testing
        test_col=test_targ[column].to_numpy()
    
    
        # Target values from chosen column that we will train with
        train_col=train_targ[column].to_numpy()
    
        history = model.fit(train_feat, train_col, validation_split=0.2, epochs=200, verbose=False)
        results = model.evaluate(test_feat, test_col)
        # print(targ_name, 'NN MSE:', results)
        # print(targ_name, "NN Average MSE over last 50 points: ",
        #       np.average(history.history['val_loss'][-50:]))
        # plt.plot(history.history['val_loss'])
        # plt.show()
    
        # find r squared
        y_pred= model.predict(test_feat)
    
        r2= r2_score(test_col, y_pred)
        NN_data.append(r2)
    
    
    NNdf=pd.DataFrame(data=NN_data, index=[targets.columns], dtype=None, copy=False)

    print(NNdf)

    
    

#%% Models
if just_NN==False:
    features='Full Text Embeddings'
    
    data=[]
    for column in test_targ:
        row=[]
        targ_name= column
        targ_name=targ_name.replace('_', ' ')
        
        # Target value from specific column that we'll use for testing
        test_col=test_targ[column]
        
        
        # Target values from chosen column that we will train with
        train_col=train_targ[column]
        
        #GB takes in 1d arrays for y
        
        
        reg = LinearRegression().fit(train_feat, train_col)
        
        test_score= reg.score(test_feat, test_col)
        train_score=reg.score(train_feat, train_col)
        print("{} Linear regression R^2 test score: ".format(targ_name),test_score )
        # print("{} Linear regression R^2 train score: ".format(targ_name), train_score )
        
        lr_preds=reg.predict(test_feat)
        
        row.append(test_score)
        
        
        plt.plot(test_col,lr_preds,'ro',ms=4)
        plt.ylabel('Predicted')
        plt.xlabel('Actual')
        title= '{} Linear Regression Actual v. Predicted using {}'.format(targ_name, features)
        plt.ylim(0,6)
        # draw diagonal line from (0,0) to (6,6)
        plt.plot([0,6], [0, 6], 'k-')
        plt.title(title)
        if plot:
            plt.savefig('Plots/{}/{}.png'.format(folder,title))
        plt.show()
        
        #%% Gradient Boosting Regressor
        gb = GradientBoostingRegressor(random_state=0)
        gb.fit(train_feat,train_col)
        
        score=gb.score(test_feat,test_col)
        gb_preds= gb.predict(test_feat)  
        
        
        print("{} Gradient Boosting Regressor R^2 test score: ".format(targ_name), score)
        
        row.append(score)
        
        plt.plot(test_col,gb_preds,'bo',ms=4)
        plt.ylabel('Predicted')
        plt.xlabel('Actual')
        title= '{} Gradient Boosting Actual v. Predicted using {}'.format(targ_name, features)
        plt.title(title)
        plt.xlim(0,6)
        plt.ylim(0,6)
        # draw diagonal line from (0,0) to (6,6)
        plt.plot([0,6], [0, 6], 'k-')
        # plt.axis([0,1,0,1])
        # plt.xticks(np.arange(0, 1, 0.1))
        # plt.yticks(np.arange(0, 1,0.1))
        if plot:
            plt.savefig('Plots/{}/{}.png'.format(folder,title))
        plt.show()
        
        #%% Random Forest Regressor
        
        rf = RandomForestRegressor(random_state=0)
        rf.fit(train_feat,train_col)
        
        score=rf.score(test_feat,test_col)
        rf_preds= rf.predict(test_feat)
        
        print("{} Random Forest Regressor R^2 test score: ".format(targ_name), score)
        
        row.append(score)
        
        plt.plot(test_col,rf_preds,'go',ms=4)
        plt.ylabel('Predicted')
        plt.xlabel('Actual')
        plt.xlim(0,6)
        plt.ylim(0,6)
        title= '{} Random Forest Actual v. Predicted using {}'.format(targ_name, features)
        plt.title(title)
        # draw diagonal line from (0,0) to (6,6)
        plt.plot([0,6], [0, 6], 'k-')
        # plt.axis([0,1,0,1])
        # plt.xticks(np.arange(0, 1, 0.1))
        # plt.yticks(np.arange(0, 1,0.1))
        if plot:
            plt.savefig('Plots/{}/{}.png'.format(folder,title))
        plt.show()
    #%% NN
        if NN:
            test_col=test_targ[column].to_numpy()
            
            
              # Target values from chosen column that we will train with
            train_col=train_targ[column].to_numpy() 
            
            history = model.fit(train_feat, train_col, validation_split=0.2, epochs=200, verbose=False)
            results = model.evaluate(test_feat, test_col)
            # print(targ_name, 'NN MSE:', results)
            # print(targ_name, "NN Average MSE over last 50 points: ",
            #       np.average(history.history['val_loss'][-50:]))
            # plt.plot(history.history['val_loss'])
            # plt.show()
            
            # find r squared
            y_pred= model.predict(test_feat)
            
            r2= r2_score(test_col, y_pred)
            print("{} Neural Net R^2 test score: ".format(targ_name), r2)
            row.append(r2)
            plt.plot(test_col,y_pred, 'co',ms=4)
            plt.ylabel('Predicted')
            plt.xlabel('Actual')
            plt.xlim(0,6)
            plt.ylim(0,6)
            title= '{} NN Actual v. Predicted using {}'.format(targ_name, features)
            plt.title(title)
            # draw diagonal line from (0,0) to (6,6)
            plt.plot([0,6], [0, 6], 'k-')
            # plt.axis([0,1,0,1])
            # plt.xticks(np.arange(0, 1, 0.1))
            # plt.yticks(np.arange(0, 1,0.1))
            if plot:
                plt.savefig('Plots/{}/{}.png'.format(folder,title))
            plt.show()
        
        data.append(row)
    
    if NN:
        df=pd.DataFrame(data=data, index=[targets.columns], columns=['Linear Regression', 'Gradient Boosting','Random Forest', 'Neural Net'], dtype=None, copy=False)
    else: 
        df=pd.DataFrame(data=data, index=[targets.columns], columns=['Linear Regression', 'Gradient Boosting','Random Forest'], dtype=None, copy=False)            
#%%   Df to excel
if to_excel:
    if pca_red:
        df.to_excel('Exported Excel/pca_both_model_{}.xlsx'.format(folder))
    else: 
        df.to_excel('Exported Excel/no_text_model_{}.xlsx'.format(folder))
        
#%% RF Plot
targ_name= 'Useful_Average'
targ_name=targ_name.replace('_', ' ')

# Target value from specific column that we'll use for testing
test_col=test_targ['Useful_Average']


# Target values from chosen column that we will train with
train_col=train_targ['Useful_Average']

rf = RandomForestRegressor(random_state=0)
rf.fit(train_feat,train_col)
        
score=rf.score(test_feat,test_col)
rf_preds= rf.predict(test_feat)

print("{} Random Forest Regressor R^2 test score: ".format(targ_name), score)

row.append(score)

#%%
plt.scatter(test_col,rf_preds,c='#004c6d')
plt.ylabel('Predicted Rating')
plt.xlabel('Actual Rating')
plt.xlim(0,8)
plt.ylim(0,8)
title= 'Random Forest Actual v. Predicted Usefulness'
plt.title(title)
# draw diagonal line from (0,0) to (6,6)
lines=plt.plot([1,7], [1, 7])
plt.setp(lines, color='lightgrey', ls='--')
plt.gca().set_aspect('equal', adjustable='box')
# plt.axis([0,1,0,1])
# plt.xticks(np.arange(0, 1, 0.1))
# plt.yticks(np.arange(0, 1,0.1))
if plot:
    plt.savefig('Plots/{}/{}.png'.format(folder,title))
plt.show()