# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 14:14:34 2021

@author: krist
"""
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

features = 'all_nov'

one_hot=np.loadtxt('{}.txt'.format(features))

# text_features=np.loadtxt('text_embeddings.txt')
full_embeddings=pd.read_excel('full_text_embeddings.xlsx',engine='openpyxl')


text_desc=pd.read_excel('Milk Frother rating descriptions edited.xlsx', engine='openpyxl')
desc_embedding=pd.read_excel('text_embeddings.xlsx', engine='openpyxl')
#%% Remove NaNs
# Row 753
full_embeddings.drop('Unnamed: 0', axis=1, inplace=True)
desc_embedding.drop('Unnamed: 0', axis=1, inplace=True)

text_desc=text_desc.drop(753, axis=0)
full_embeddings=full_embeddings.drop(753, axis=0)
one_hot=np.delete(one_hot, (753), axis=0)
desc_embedding=desc_embedding.drop(753, axis=0, inplace=True)
#%% The rows of one_hot are each a model, must compare them
#cattle=5
#bike =8
#rodeo=116
d1=8
d2=116

descs=text_desc['Idea description']

cattle=descs[d1]
bicycle= descs[d2]


cattle_text=desc_embedding.loc[d1, :].to_numpy()
bike_text=desc_embedding.loc[d2, :].to_numpy()


cattle_one_hot=one_hot[d1:d1+1, :]
bike_one_hot=one_hot[d2:d2+1,:]

cattle_full= full_embeddings.loc[d1, :].to_numpy()
bike_full= full_embeddings.loc[d2, :].to_numpy()

cos_sim= cosine_similarity(cattle_one_hot, bike_one_hot)
text_sim= cosine_similarity(np.array([cattle_text]), np.array([bike_text]))
full_sim=cosine_similarity(np.array([cattle_full]), np.array([bike_full]))


print('One hot similarity: ', cos_sim )
print('Text similarity: ', text_sim )
print('Full similarity: ', full_sim )

#%%
def plot_similarity(labels, features, rotation):
  corr = np.inner(features, features)
  sns.set(font_scale=1.2)
  g = sns.heatmap(
      corr,
      xticklabels=labels,
      yticklabels=labels,
      vmin=0,
      vmax=1,
      cmap="YlOrRd")
  g.set_xticklabels(labels, rotation=rotation)
  g.set_title("Semantic Textual Similarity")
  
def run_and_plot(messages_):
  message_embeddings_ = embed(messages_)
  plot_similarity(messages_, message_embeddings_, 90)
  
#%%
import matplotlib.pyplot as plt

index = ['Bicylce', 'Cattle', 'Rodeo']
columns = ['Bicylce', 'Cattle', 'Rodeo']

data=np.array([[1,0.58925565,0.333333],[0.58925565, 1, 0.23570226],[0.33333333,0.23570226,1]])
df = pd.DataFrame(data=data, index=index, columns=columns)

plt.pcolor(df)
plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
plt.show()