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
    tokenizer='IsmatS/xlm-roberta-az-ner'
)

başlıq_nəticələri = başlıq_nlp(df['başlıq'].tolist())
for başlıq_entities in başlıq_nəticələri:
    for entity in başlıq_entities:
        print(f"{entity['word']} -> {entity['entity']}") 