#!/usr/bin/python

import pandas as pd
import numpy as np
import sys
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

def predict(txt):
    clf = pickle.load(open('clf', 'rb'))
    # Load the vocabulary
    vocabulary_to_load = pickle.load(open('dict', 'rb'))
    vect = CountVectorizer(max_features=512, vocabulary = vocabulary_to_load)
    X = vect.transform([txt])
    res = clf.predict_proba(X)
    
    cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
            'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
            'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']
    
    return dict(zip(cols, res[0]))


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add data')
        
    else:

        txt      = sys.argv[1]

        p1 = predict(txt)
        
        #print(url)
        print('predictions: ', p1)
        