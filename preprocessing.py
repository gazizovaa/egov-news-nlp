import numpy as np 
import pandas as pd 
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from azstemmer import AzStemmer

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




