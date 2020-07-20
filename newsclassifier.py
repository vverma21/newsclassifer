# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 00:09:19 2019

@author: VISHESH
"""

import sklearn.datasets as skd 
categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
news_train = skd.load_files('E:/20news-bydate-train', categories= categories, encoding= 'ISO-8859-1')

news_test = skd.load_files('E:/20news-bydate-test',categories= categories, encoding= 'ISO-8859-1')

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
text_clf = Pipeline([('vect', TfidfVectorizer()),('clf', MultinomialNB()) ])
text_clf.fit(news_train.data, news_train.target)
predicted = text_clf.predict(news_test.data)
from sklearn import metrics
from sklearn.metrics import accuracy_score
import numpy as np
print('Accuracy achieved is ' + str(np.mean(predicted == news_test.target)))
print(metrics.classification_report(news_test.target, predicted, target_names=news_test.target_names)),

