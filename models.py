import preprocess_and_feature_extract
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# xəbər başlıqları məlumatını feature-target olaraq ayırırıq
X_başlıq = np.array(preprocess_and_feature_extract.df_cleaned['başlıq_vektor'].tolist())
y = preprocess_and_feature_extract.df_cleaned['baxış_sayı']
# train-test olaraq ayırırıq
X_train, X_test, y_train, y_test = train_test_split(X_başlıq, y, test_size=0.2, random_state=42)
# baseline model qururuq
başlıq_model = LinearRegression()
başlıq_model.fit(X_train, y_train)
print(başlıq_model.score(X_test, y_test))

# xəbər məzmunları məlumatını feature-target olaraq ayırırq
X_məzmun = np.array(preprocess_and_feature_extract.df_cleaned['məzmun_vektor'].tolist())
y = preprocess_and_feature_extract.df_cleaned['baxış_sayı']
# train-test olaraq ayırırıq
X_train, X_test, y_train, y_test = train_test_split(X_məzmun, y, test_size=0.2, random_state=42)
# baseline model qururuq
məzmun_model = LinearRegression()
məzmun_model.fit(X_train, y_train)
print(məzmun_model.score(X_test, y_test))

