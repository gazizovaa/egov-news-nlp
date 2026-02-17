import numpy as np 
import pandas as pd 
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

df = pd.read_csv('egov_news.csv')
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
# ingilis dilindəki stopwords-ləri əldə edirik
stopwords_list = set(stopwords.words('english'))

# silinmə prosesi
df_cleaned['title'] = df_cleaned['title'].apply(
    lambda row: [word for word in word_tokenize(row) if word not in stopwords_list])
print(df_cleaned['title'])

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
print(df_cleaned['content'])

# Stemming 
