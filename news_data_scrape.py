from bs4 import BeautifulSoup 
import requests
import time 
import pandas as pd 

all_pages = 83
url = "https://www.e-gov.az/az/news"
news_data = []

for page_numb in range(1, all_pages + 1):
    # the preparation of http request - this prevents the site from preventing the program as a bot and block it
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    
    # the url sends a request to http
    response = requests.get(url, headers=headers) 
    
    if response.status_code == 200:
        print(f"Page {page_numb} loaded successfully!")
        
        # create a BeautifulSoup object and specify the parser
        soup = BeautifulSoup(response.content, "html.parser")
        news_list = soup.find('ul', class_='news-list')
        
        # find all <li> tags 
        items = news_list.find_all('li')
        for item in items:
            try:
                # find the <a> tag and read the title attribute
                link = item.find('a')
                
                # get the news headline
                title = link.get('title').strip()
                
                # get the url address of each news
                base_url = "https://www.e-gov.az"
                url_address = base_url + link.get('href', '').strip()
                
                # get the news content
                news_soup = BeautifulSoup(requests.get(url_address, headers=headers).content, 
                                          "html.parser")
                content = ""
                
                for p in news_soup.select('#NewsContent p'):
                    content += p.text.strip() + " "
                
                # get the published date
                tools_div = item.select_one('div.tools')
                time_tag = tools_div.find('time')
                published_date = time_tag.text.strip()
                
                # get the view statistics
                views_div = tools_div.find_all('div')[-1]
                views_count = views_div.text.strip()
                
                news_data.append({'title': title,
                                  'url': url_address,
                                  'content': content,
                                  'published date': published_date,
                                  'views': views_count})  
            except Exception as e:
                print(f"Obtained an error: {e}")
                continue 
        time.sleep(2)
   
df = pd.DataFrame(data=news_data)
df.to_csv('egov_news.csv', encoding='utf-8')
