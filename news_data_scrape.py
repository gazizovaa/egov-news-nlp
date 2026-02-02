from bs4 import BeautifulSoup 
import requests
import time 
import pandas as pd 

all_pages = 83
url = "https://www.e-gov.az/az/news"
news_data = []

for page_numb in range(1, all_pages + 1):
    # Add user-agent to the website in order not be considered as a bot
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    # Get a response from te site
    response = requests.get(url, headers=headers) 
    
    if response.status_code == 200:
        print(f"Page {page_numb} loaded successfully!")
        
        # Create a BeautifulSoup object and specify the parser
        soup = BeautifulSoup(response.content, "html.parser")
        news_list = soup.find('ul', class_='news-list')
        
        # Find all <li> tags 
        items = news_list.find_all('li')
        for item in items:
            try:
                # Find the <a> tag and read the title attribute
                link = item.find('a')
                if link and link.get('title'):
                    title = link.get('title').strip()
                news_data.append({'title': title}) 
            except Exception as e:
                print(f"Obtained an error: {e}")
                continue 
        time.sleep(2)
   
df = pd.DataFrame(data=news_data)
df.to_csv('egov_news.csv', encoding='utf-8')
