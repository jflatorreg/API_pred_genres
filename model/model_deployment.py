#!/usr/bin/python

import pandas as pd
import numpy as np
import sys

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from keras.activations import relu, swish, sigmoid
from sentence_transformers import SentenceTransformer

def predict(txt):
     
    def wider_model():
        model = Sequential()
        model.add(Dense(669, input_shape=(768,),activation=relu))
        model.add(Dropout(0.10769379659079174))
        model.add(Dense(96, input_shape=(768,),activation=swish))
        model.add(Dropout(0.022278422538476006))
        model.add(Dense(24, activation=sigmoid))
        return model

    model = wider_model()
    model.load_weights("./model_1.h5")
    
    data = pd.DataFrame([txt], columns= ['plot'])
    Transformer = SentenceTransformer('bert-base-uncased')
    sentence_embeddings = Transformer.encode(data['plot'].values)
    pre = model.predict(sentence_embeddings)
    cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']
    res = pd.DataFrame(pre, index=data.index, columns=cols)
    
    return str(res.to_dict('records')[0])


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add data')
        
    else:

        txt      = sys.argv[1]

        p1 = predict(txt)
        
        #print(url)
        print('price: ', p1)
        