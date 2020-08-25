import  warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from stop_words import get_stop_words
import nltk, string
from nltk.stem.porter import PorterStemmer

class Preprocessing:
    def __init__(self):
        self.porter = PorterStemmer()
        self.stop_words = get_stop_words('en')
        self.stop_words.append(' ')
        self.stop_words.append('')
    
    def pipeline(self, df):
        for lang in ['description']:
            #, 'translate_es', 'translate_fr', 'translate_de', 'translate_ja']:
            df[lang] = df[lang].apply(lambda x: self.change_text(x))
        return df

    def change_text(self, text):
        text = text.lower()
        text = text.replace('ml', 'machine learning')
        text = "".join([char if char not in string.punctuation else ' ' for char in text])
        text = " ".join([self.porter.stem(char) for char in text.split(' ') if char not in self.stop_words])
        return text
    
    def vectorize_tfidf(self, df):
        vec_tfidf = TfidfVectorizer()
        X = vec_tfidf.fit_transform(df.values)
        X = pd.DataFrame(X.toarray(), columns=vec_tfidf.get_feature_names())
        return X
    
    def vectorize_cnt(self, df):
        vec_cnt = CountVectorizer()
        X = vec_cnt.fit_transform(df.values)
        X = pd.DataFrame(X.toarray(), columns=vec_cnt.get_feature_names())
        return X

def make_tfidf_df(o_df, col='description'):
    df = o_df.copy()
    id_cols = ['jobflag','text_id']
    preprocessing = Preprocessing()
    df[col] = df[col].apply(lambda x: preprocessing.change_text(x))
    X = preprocessing.vectorize_tfidf(df[col])
    X = pd.concat([df[id_cols], X], axis=1)
    train_df = X[X.jobflag.notnull()].reset_index(drop=True)
    test_df = X[X.jobflag.isnull()].drop(columns=['jobflag']).reset_index(drop=True)
    mlp_feature = train_df.drop(columns=id_cols).columns.tolist()
    return train_df, test_df, mlp_feature

def make_vec_cnt_df(o_df, col='description'):
    df = o_df.copy()
    id_cols = ['jobflag','text_id']
    preprocessing = Preprocessing()
    langs = ['description', 'translate_de', 'translate_es', 'translate_fr', 'translate_ja']
    for lan_col in langs:
        df[lan_col] = df[lan_col].apply(lambda x: preprocessing.change_text(x))
        
    X = preprocessing.vectorize_cnt(df[col])
    X = pd.concat([df[id_cols], X], axis=1)
    train_df = X[X.jobflag.notnull()].reset_index(drop=True)
    test_df = X[X.jobflag.isnull()].drop(columns=['jobflag']).reset_index(drop=True)
    mlp_feature = train_df.drop(columns=id_cols).columns.tolist()
    return train_df, test_df, mlp_feature