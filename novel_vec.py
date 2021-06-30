
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity




if __name__== "__main__":
    
    """
    This code converts Shah vectors to one-hot encoding 


    """    
    
    #novelty.txt rows have SVS vectors
    novelty=np.loadtxt('novelty_no_text.txt')
    num_ideas,num_col=np.shape(novelty)
    all_nov=np.array([], dtype=np.int64).reshape(num_ideas,0)

    for i in range(num_col):
        vec=novelty[:,i]
        tempmatr=np.zeros((num_ideas,int(np.max(vec))))
        for j in range(num_ideas):
            if(novelty[j,i]>0):
                print([novelty[j,i], j])
                tempmatr[j,int(novelty[j,i])-1]=1
        all_nov=np.concatenate((all_nov,tempmatr),axis=1)
    
 
    #Columns to remove
    remov_col= np.all(all_nov == 0, axis=0)
    
    #One hot encoded feature matrix
    matr=all_nov[:,~remov_col]
    
    #Cosine similarity between sketches
    simil=cosine_similarity(matr,matr)
