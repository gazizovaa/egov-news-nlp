import pandas as pd 
from transformers import pipeline

df = pd.read_csv('data/egov_news.csv')

df.rename(columns={'title': 'başlıq', 'content': 'məzmun',
                   'published date': 'dərc_olunma_tarixi',
                   'views': 'baxış_sayı'}, inplace=True)

df.drop(columns=['Unnamed: 0', 'url'], inplace=True)

# pip install transformers torch
başlıq_nlp = pipeline(
    task='ner',
    model='IsmatS/xlm-roberta-az-ner',
    tokenizer='IsmatS/xlm-roberta-az-ner',
    aggregation_strategy='simple' # token-ları birləşdirir
)

başlıq_nəticələri = başlıq_nlp(df['başlıq'].tolist())

for i, text_results in enumerate(başlıq_nəticələri):
    print(f"Xəbər başlığı {i+1}")
    for entity in text_results:
        print(f"Entity: {entity['word']}, Label: {entity['entity_group']}") 