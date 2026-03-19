import numpy as np 
import pandas as pd 
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from transformers import pipeline
from azstemmer import AzStemmer
import simplemma

df = pd.read_csv('data/egov_news.csv')
pd.set_option('display.max_columns', None) 

# oxunaqlılıq üçün bəzi sütun adlarını dəyişdiririk
df.rename(columns={'title': 'başlıq', 'content': 'məzmun',
                   'published date': 'dərc_olunma_tarixi',
                   'views': 'baxış_sayı'}, inplace=True)
df.drop(columns=['Unnamed: 0'], inplace=True)

# dərc_olunma_tarixi sütunun tipinin dəyişilməsi
df['dərc_olunma_tarixi'] = pd.to_datetime(df['dərc_olunma_tarixi'], format='%d.%m.%Y', errors='coerce')
df.dropna(subset=['dərc_olunma_tarixi'], inplace=True)  

# dərc_olunma_tarixi sütununa feature engineering-in tətbiqi
df['dərc_olunma_ili'] = df['dərc_olunma_tarixi'].dt.year
df['dərc_olunma_ayı'] = df['dərc_olunma_tarixi'].dt.month
df['dərc_olunma_günü'] = df['dərc_olunma_tarixi'].dt.day
df['dərc_olunma_həftəsi'] = df['dərc_olunma_tarixi'].dt.dayofweek + 1

# # dərc_olunma_tarixi sütununun silinməsi
df.drop(columns=['dərc_olunma_tarixi'], inplace=True)

# # il, ay və günə görə sıralama 
df_cleaned = df.sort_values(by=['dərc_olunma_ili']).reset_index(drop=True)
# print(df_cleaned) 

#####Text Normalization#####
print("-----Text Normalization-----")
# 'başlıq' və 'məzmun' sütunlarını daşıyan mətnləri kiçik hərflərlə yazılan mətnlərə çevrilməsi
df_cleaned['başlıq'] = df_cleaned['başlıq'].str.lower()
df_cleaned['məzmun'] = df_cleaned['məzmun'].str.lower()

# rəqəmlərin silinməsi
df_cleaned['başlıq'] = df_cleaned['başlıq'].str.replace(r'\d+', '', regex=True)
df_cleaned['məzmun'] = df_cleaned['məzmun'].str.replace(r'\d+', '', regex=True)

# durğu işarələrinin silinməsi
df_cleaned['başlıq'] = df_cleaned['başlıq'].str.replace(r'[^\w\s]', '', regex=True)
df_cleaned['məzmun'] = df_cleaned['məzmun'].str.replace(r'[^\w\s]', '', regex=True)

# NLTK ilə Tokenization
# punkt_tab modelini yükləyirik -> nltk kitabxanasının pre-trained olunmuş modelidir
nltk.download("punkt_tab")

# stopwords removal tətbiqi
nltk.download("stopwords")

# azərbaycan dilindəki stopwords-ləri əldə edilməsi və silinməsi
stopwords_lst = set(stopwords.words('azerbaijani'))
# print(stopwords_lst) 
df_cleaned['başlıq'] = df_cleaned['başlıq'].apply(
    lambda row: [word for word in word_tokenize(row) if word not in stopwords_lst]
    if isinstance(row, str) else [])
print(df_cleaned['başlıq'])

# xəbər başlıq cümlələrini sözlərə parçalanması
for row in df_cleaned['başlıq']:
    print(row)

# xəbər kontentlərinə daxil olan cümlələrin sözlərə parçalanması
# stopwords removal tətbiqi
def tokenize_content(text):
    if pd.isna(text):
        return []
    
    sentences = sent_tokenize(text)
    words_list = []
    for sentence in sentences:
        words_list.extend(word_tokenize(sentence))
    return [word for word in words_list if word not in stopwords_lst]

df_cleaned['məzmun'] = df_cleaned['məzmun'].apply(tokenize_content)

# Stemming 
stemmer = AzStemmer(keyboard='az')

df_cleaned['başlıq_stem'] = df_cleaned['başlıq'].apply(
    lambda tokens: [stemmer.stem(word) for word in tokens]
)

df_cleaned['məzmun_stem'] = df_cleaned['məzmun'].apply(
    lambda tokens: [stemmer.stem(word) for word in tokens]
)
print(df_cleaned[['başlıq_stem', 'məzmun_stem']])

# token listləri yenidən string-ə convert edirik
df_cleaned['başlıq'] = df_cleaned['başlıq'].apply(lambda x: ' '.join(x))
df_cleaned['məzmun'] = df_cleaned['məzmun'].apply(lambda x: ' '.join(x))

#####Feature Extraction#####
print('-----Feature Extraction-----')
# Bag-Of-Words 
başlıq_vectorizer = CountVectorizer(max_features=1000)
başlıq_bow = başlıq_vectorizer.fit_transform(df_cleaned['başlıq'])
print(f"Başlıqlar üçün Count Vectorizer texnikası:\n{başlıq_bow}") 

məzmun_vectorizer = CountVectorizer(max_features=1000)
məzmun_bow = məzmun_vectorizer.fit_transform(df_cleaned['məzmun'])
print(f"Məzmunlar üçün count Vectorizer texnikası:\n{məzmun_bow}") 

# TF-IDF
başlıq_tfidf_vectorizer = TfidfVectorizer(max_features=1000)
başlıq_tfidf = başlıq_tfidf_vectorizer.fit_transform(df_cleaned['başlıq'])
print(f"Başlıqlar üçün TF-İDF texnikası:\n{başlıq_tfidf}") 

məzmun_tfidf_vectorizer = TfidfVectorizer(max_features=1000)
məzmun_tfidf = məzmun_tfidf_vectorizer.fit_transform(df_cleaned['məzmun'])
print(f"Məzmunlar üçün count Vectorizer texnikası:\n{məzmun_bow}") 

# WordVec
başlıq_wordvec = Word2Vec(sentences=df_cleaned['başlıq_stem'], 
                          vector_size=100, 
                          window=3, 
                          min_count=1)

məzmun_wordvec = Word2Vec(sentences=df_cleaned['məzmun_stem'],
                          vector_size=100,
                          window=5,
                          min_count=2)

# hər bir sözü vektorlara çevirir
# bütün vektorların ortalamasını hesablayır 
# ən sonda 1 vektor yaradır
def sentence_vector(tokens, model):
    if not isinstance(tokens, list) or len(tokens) == 0:
        return np.zeros(model.vector_size)
    
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0) 
        

df_cleaned['başlıq_vektor'] = df_cleaned['başlıq_stem'].apply(
    lambda tokens: sentence_vector(tokens, başlıq_wordvec))

df_cleaned['məzmun_vektor'] = df_cleaned['məzmun_stem'].apply(
    lambda tokens: sentence_vector(tokens, məzmun_wordvec))









