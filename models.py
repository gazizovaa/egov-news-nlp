import preprocess_and_feature_extract
from sklearn import preprocessing
from sklearn.cluster import KMeans

X = preprocess_and_feature_extract.başlıq_tfidf
# normalizing
X_normaalized = preprocessing.normalize(X)

# dataset unlabeled olduğundan KMeans Clustering modelindən istifadə edirik
egov_kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
egov_kmeans.fit(X_normaalized)
print(egov_kmeans.score(X)) 

