import pandas as pd 
from transformers import pipeline 

df = pd.read_csv('data/egov_news.csv')

df.rename(columns={'title': 'başlıq', 'content': 'məzmun',
                   'published date': 'dərc_olunma_tarixi',
                   'views': 'baxış_sayı'}, inplace=True)

df.drop(columns=['Unnamed: 0', 'url'], inplace=True)

summarizer = pipeline(
    task='summarization',
    model='nijatzeynalov/mT5-based-azerbaijani-summarize'

)
# Xəbər kontentlərinin xülasə edilməsi
def summarize_text_with_hf(text, max_length=100, min_length=10):
    # token-lara limit qoyulması
    text = text[:100]
    result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False) 
    # do_sample=False -> model hər addımda bütün sözlərin ehtimalını hesablayır, sonra
    # ən yüksək ehtimallı sözü seçir.
    return result[0]['summary_text']

df['xülasə'] = df['məzmun'].apply(summarize_text_with_hf)
print(df[['məzmun', 'xülasə']].head())