# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:49:54 2021

@author: krist
"""

import pandas as pd
import numpy as np
import math

average=True



svs_features= pd.read_excel('SP2016Copy2.xls', sheet_name='novelty' )

meanings= pd.read_excel('SP2016Copy.xls', sheet_name='Text descriptions' )

text_desc= pd.read_excel('Milk Frother rating descriptions edited.xlsx')

#import targets from excel sheet as dataframe
#to convert dataframe to numpy, np.asarray(<dataframe>)

targets=pd.read_excel('Edited_Creativity_Metrics.xlsx', sheet_name='Sheet2')
# Pandas uses iloc to locate based on index number rather than label (loc)
# Targets are a datagrame and features are just an array

#%% If average

if average:
    folder='average vals'
    targets=pd.read_excel('Edited_Creativity_Metrics.xlsx', sheet_name='Sheet4')

#%% Remove NaNs
# Row 753

targets=targets.drop(753, axis=0)
svs_features = svs_features.drop(753, axis=0)
text_desc= text_desc.drop(753, axis=0)
# one_hot_features=np.delete(one_hot_features, (753), axis=0)

#%% Remove first two columns and after row 934 from svs_features

cols=[0,1]
svs_features.drop(svs_features.columns[cols], axis=1, inplace=True)


svs_features.drop(svs_features.index[933:], axis=0, inplace=True)

svs=svs_features[svs_features.columns[0:78]]

#%% svs is the df of svs features, meanings is the df of meanings of the values

designs=svs.index
#there are 934 designs going from 0-933 (indicates 934 rows)

# a.append('text') to add dimensions to a list
# a[i]=a[i]+'text' adds more text to an existing dimension in the list

strings=[]
for i in range(934):
    strings.append(' ')

#%%
for columns in svs:
    for ind in svs.index:
        # print('For column {} and row {}'.format(columns, ind))
        val=svs.loc[ind,columns]
        # print(val)
        if val:
            strings[ind]= strings[ind]+' '+ columns
            if columns in meanings.columns:
                desc=meanings.loc[val-1,columns]
                
                if isinstance(desc, float):
                     desc=' '
                strings[ind]= strings[ind]+' '+ desc


#%% Add in the text descriptions

for i in text_desc.index:
    text=text_desc.loc[i, 'Idea description']
    #deal with nans
    if isinstance(text, float):
                     text=' '
    strings[i]=strings[i]+ ' ' + text

#%% Remove 753 (any blank rows)
strings.remove(' ')

strings_array=np.array(strings)

df=pd.DataFrame(data=strings_array, index=text_desc.index)




















