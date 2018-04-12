import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter


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

if __name__ == '__main__':
    s = ["hello world", "this is a test"]
    y = [0, 0]
    print("Testing indirect text feature extraction")
    print("Input: ", s)
    print(NumUniqueWordExtractor().fit_transform(s, y))
    print(NumWordExtractor().fit_transform(s, y))
    print(CharLengthExtractor().fit_transform(s, y))
    print(AverageWordLengthExtractor().fit_transform(s, y))