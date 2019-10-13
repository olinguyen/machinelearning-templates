import numpy as np
import pandas as pd
import os
import re

from typing import List
from collections import defaultdict, Counter
from pprint import pprint
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import fetch_20newsgroups


class NumWordExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def transform(self, X):
        return np.array([len(sentence.split()) for sentence in X]).reshape(-1, 1)

    def fit(self, X, y):
        return self

class AverageWordLengthExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def average_word_length(self, text):
        """Helper code to compute average word length of a name"""
        return np.mean([len(word) for word in text])
    
    def transform(self, X):
        return np.array([self.average_word_length(sentence.split()) for sentence in X]).reshape(-1, 1)

    def fit(self, X, y):
        return self
    
class CharLengthExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def transform(self, X):
        return np.array([len(sentence) for sentence in X]).reshape(-1, 1)

    def fit(self, X, y):
        return self    
    
class NumUniqueWordExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def transform(self, X):
        return np.array([len(Counter(sentence.split())) for sentence in X]).reshape(-1, 1)

    def fit(self, X, y):
        return self


def main():
    categories = ['alt.atheism', 'talk.religion.misc',
		   'comp.graphics', 'sci.space']

    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))

    # Number of examples: 2034 
    pprint(list(newsgroups_train.target_names))
    print("Number of examples:", len(newsgroups_train.data))

    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

    feature_extractor = FeatureUnion([
	("tfidf_token_ngrams", TfidfVectorizer(ngram_range=(1,2),
			       lowercase=False,
			       stop_words='english')),
	("tfidf_char_ngrams", TfidfVectorizer(analyzer='char',
			       ngram_range=(1,2),
			       lowercase=False,
			       stop_words='english')),
	('num_unique_words', NumUniqueWordExtractor()),
	('char_len', CharLengthExtractor()),
	('num_words', NumWordExtractor())

    ])

    lr_tfidf = Pipeline([
		('feature_extraction', feature_extractor),
		('logistic_regression', GridSearchCV(
				LogisticRegression(penalty='l2', 
						   random_state=42), 
						   param_grid=params))])

    X: List[str] = newsgroups_train.data
    y: List[int] = newsgroups_train.target

    scores = cross_val_score(lr_tfidf, X, y, cv=2, n_jobs=-1)
    print(scores)


if __name__ == "__main__":
    main()
