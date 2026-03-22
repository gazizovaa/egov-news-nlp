from sklearn.compose import ColumnTransformer
import preprocessing
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# "baxış_sayı" sütunun 3 bərabər alt-kateqoriyaya bölünməsi
preprocessing.df_cleaned['baxış_sinifləri'] = pd.qcut(
    preprocessing.df_cleaned['baxış_sayı'],
    q=3,
    labels=['aşağı', 'orta', 'yuxarı']
)
print(preprocessing.df_cleaned['baxış_sinifləri'].value_counts())

# stem sütunlarını çıxardıb ayrıca saxlayırıq (Word2Vec metodu üçün)
stem_data = preprocessing.df_cleaned[['başlıq_stem', 'məzmun_stem']]

# X və y-in ayrılması
X = preprocessing.df_cleaned.drop(['url', 'baxış_sinifləri', 'baxış_sayı',
                                   'başlıq_stem', 'məzmun_stem'], axis=1)
y = preprocessing.df_cleaned['baxış_sinifləri'].copy()

# train-test splitting prosesinin icra edilməsi
X_full_train, X_test, stem_full_train, stem_test, y_full_train, y_test = train_test_split(X, stem_data, y, test_size=0.2, random_state=42)
X_train, X_val, stem_train, stem_val, y_train, y_val = train_test_split(X_full_train, stem_full_train, y_full_train, test_size=0.2, random_state=42)

# rəqəmsal sütunların çıxarılması
num_features = X_train.select_dtypes(include=[np.number]).columns 

# rəqəmsal sütunlar üçün pipeline-nın yaradılması
num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# mətn pipeline-larının qurulması
başlıq_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000))
])

məzmun_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000))
])

transformer = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('başlıq', başlıq_pipeline, 'başlıq'),
    ('məzmun', məzmun_pipeline, 'məzmun')
], remainder='passthrough')

# training setinin fit_transform, digərlərinin sadəcə transform edilməsi
X_train_final = transformer.fit_transform(X_train)
X_val_final = transformer.transform(X_val)
X_test_final = transformer.transform(X_test)

# Word2Vec metodunun fit edilməsi
başlıq_w2v = Word2Vec(sentences=stem_train['başlıq_stem'], vector_size=100, window=3, 
                      min_count=1)
məzmun_w2v = Word2Vec(sentences=stem_train['məzmun_stem'], vector_size=100, window=5, 
                      min_count=2)

def sentence_vector(tokens, model):
    if not isinstance(tokens, list) or len(tokens) == 0:
        return np.zeros(model.vector_size)
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

def get_w2v_matrix(stem_col, model):
    return np.vstack(stem_col.apply(lambda t: sentence_vector(t, model)))

# Word2Vec matrislərini dense matrislərinə çevrilməsi
train_başlıq_w2v = get_w2v_matrix(stem_train['başlıq_stem'], başlıq_w2v)
val_başlıq_w2v = get_w2v_matrix(stem_val['başlıq_stem'], başlıq_w2v)
test_başlıq_w2v = get_w2v_matrix(stem_test['başlıq_stem'], başlıq_w2v)

train_məzmun_w2v = get_w2v_matrix(stem_train['məzmun_stem'], məzmun_w2v)
val_məzmun_w2v = get_w2v_matrix(stem_val['məzmun_stem'], məzmun_w2v)
test_məzmun_w2v = get_w2v_matrix(stem_test['məzmun_stem'], məzmun_w2v)

# Transformerin çıxışını dense-ə çeviririk
X_train_dense = X_train_final.toarray()
X_val_dense = X_val_final.toarray()
X_test_dense = X_test_final.toarray()

# hamısını birləşdiririk
X_train_final = np.concatenate([X_train_dense, train_başlıq_w2v, train_məzmun_w2v], axis=1)
X_val_final = np.concatenate([X_val_dense, val_başlıq_w2v, val_məzmun_w2v], axis=1)
X_test_final = np.concatenate([X_test_dense, test_başlıq_w2v, test_məzmun_w2v], axis=1)

