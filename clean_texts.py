import numpy as np 
import pandas as pd 
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec


df = pd.read_csv('data/egov_news.csv')
pd.set_option('display.max_columns', None) 

# oxunaqlılıq üçün bəzi sütun adlarını dəyişdiririk
df.rename(columns={'published date': 'published_date','views': 'view_count'}, inplace=True)
df.drop(columns=['Unnamed: 0'], inplace=True)

# published_date sütunun tipinin dəyişilməsi
df['published_date'] = pd.to_datetime(df['published_date'], format='%d.%m.%Y', errors='coerce')
df.dropna(subset=['published_date'], inplace=True)  

# published_date sütununa feature engineering-in tətbiqi
df['year'] = df['published_date'].dt.year
df['month'] = df['published_date'].dt.month
df['day'] = df['published_date'].dt.day
df['day_of_week'] = df['published_date'].dt.dayofweek + 1

# published_date sütununun silinməsi
df.drop(columns=['published_date'], inplace=True)

# il, ay və günə görə sıralama 
df_cleaned = df.sort_values(by=['year']).reset_index(drop=True)

# Text Normalization
# title və content sütunlarını daşıyan mətnləri kiçik hərflərlə yazılan mətnlərə çevrilməsi
df_cleaned['title'] = df_cleaned['title'].str.lower()
df_cleaned['content'] = df_cleaned['content'].str.lower()

# məna kəsb etməyən rəqəmlərin və xüsusi simvolların silinməsi
df_cleaned['title'] = df_cleaned['title'].str.replace(r'\d+', '', regex=True)
df_cleaned['content'] = df_cleaned['content'].str.replace(r'\d+', '', regex=True)

df_cleaned['title'] = df_cleaned['title'].str.replace(r'[^\w\s]', '', regex=True)
df_cleaned['content'] = df_cleaned['content'].str.replace(r'[^\w\s]', '', regex=True)

# boşluqların silinməsi
df_cleaned['title'] = df_cleaned['title'].str.replace(r'\s+', ' ', regex=True).str.strip()
df_cleaned['content'] = df_cleaned['content'].str.replace(r'\s+', ' ', regex=True).str.strip()

# NLTK ilə Tokenization
# punkt_tab modelini yükləyirik -> nltk kitabxanasının pre-trained olunmuş modelidir
# nltk.download("punkt_tab")

# xəbər başlıq cümlələrini sözlərə parçalanması
for row in df_cleaned['title']:
    print(word_tokenize(row))

# stopwords removal tətbiqi
# nltk.download("stopwords")
# azərbaycan dilindəki stopwords-ləri əldə edirik
stopwords_list = set(stopwords.words('azerbaijani'))

# silinmə prosesi
df_cleaned['title'] = df_cleaned['title'].apply(
    lambda row: [word for word in word_tokenize(row) if word not in stopwords_list])
# print(df_cleaned['title'])

# xəbər kontentlərinə daxil olan cümlələrin sözlərə parçalanması
# stopwords removal tətbiqi
def tokenize_content(text):
    if pd.isna(text):
        return []
    
    sentences = sent_tokenize(text)
    words_list = []
    for sentence in sentences:
        words_list.extend(word_tokenize(sentence))
    return [word for word in words_list if word not in stopwords_list]

df_cleaned['content'] = df_cleaned['content'].apply(tokenize_content)
# print(df_cleaned['content'])

# Stemming, Lemmetization -> the azerbaijani language isn't supported

# token listləri yenidən string-ə convert edirik
df_cleaned['title'] = df_cleaned['title'].apply(lambda x: ' '.join(x))
df_cleaned['content'] = df_cleaned['content'].apply(lambda x: ' '.join(x))

# Bag-Of-Words 
title_vectorizer = CountVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    min_df=5
)
title_bow = title_vectorizer.fit_transform(df_cleaned['title'])
print(title_vectorizer.get_feature_names_out())

content_vectorizer = CountVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    min_df=5
)
content_bow = content_vectorizer.fit_transform(df_cleaned['content'])
print(content_vectorizer.get_feature_names_out())

# ən çox istifadə olunan sözlərin tapılması
# title üçün
sum_words = title_bow.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in title_vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
title_word_freq = pd.DataFrame(words_freq, columns=['Word', 'Frequency'])
print(title_word_freq)

# content üçün
sum_words = content_bow.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in content_vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
content_word_freq = pd.DataFrame(words_freq, columns=['Word', 'Frequency'])
print(content_word_freq)

# word embedding (vectorization)
# TF-IDF
title_tfidf_vec = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8
)
title_tfidf = title_tfidf_vec.fit_transform(df_cleaned['title'])
# ən yüksək çəkisi olan sözlərin qaytarılması
title_feature_names = title_tfidf_vec.get_feature_names_out()
title_dense = title_tfidf[0].todense().tolist()[0]
title_tfidf_scores = list(zip(title_feature_names, title_dense))
print(title_tfidf.shape)
print(sorted(title_tfidf_scores, key=lambda x: x[1], reverse=True)[:5])


content_tfidf_vec = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8
)
content_tfidf = content_tfidf_vec.fit_transform(df_cleaned['content'])
# ən yüksək çəkisi olan sözlərin qaytarılması
content_feature_names = content_tfidf_vec.get_feature_names_out()
content_dense = content_tfidf[0].todense().tolist()[0]
content_tfidf_scores = list(zip(content_feature_names, content_dense))
print(content_tfidf.shape)
print(sorted(content_tfidf_scores, key=lambda x: x[1], reverse=True)[:5])

# Word2Vec embedding
title_tokens = df_cleaned['title'].tolist()
content_tokens = df_cleaned['content'].tolist()

title_w2v_model = Word2Vec(
    sentences=title_tokens,
    vector_size=100,
    window=5,
    min_count=3,
    epochs=10
)
# Modelin yadda saxlanması
title_w2v_model.save('models/title_word2vec_model')
print(f"Vocabulary Size: {len(title_w2v_model.wv)}")

content_w2v_model = Word2Vec(
    sentences=content_tokens,
    vector_size=100,
    window=5,
    min_count=3,
    epochs=10
)
# Modelin yadda saxlanması
content_w2v_model.save('models/content_w2v_model')
print(f"Vocabulary Size: {len(content_w2v_model.wv)}")



