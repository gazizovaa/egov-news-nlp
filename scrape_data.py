from bs4 import BeautifulSoup 
import requests
import time 
import pandas as pd 

# scrape ediləcək səhifələrin ümumi sayını göstərir
all_pages = 83

# xəbər keçidlərini tamamlamaq üçün əsas domen
base_url = "https://www.e-gov.az"

# veb səhifədən çıxarılacaq məlumatları yaradılmış boş listdə toplanır
news_data = []

for page_numb in range(1, all_pages + 1):
    # xəbərlərin siyahısı olan səhifə
    url = f"https://www.e-gov.az/az/news/index?page={page_numb}"

    # sayta özümüzü brauzer kimi təqdim etmək üçün başlıq
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    
    # saytın məzmununu yükləyir 
    response = requests.get(url, headers=headers) 
    
    if response.status_code == 200:
        print(f"Page {page_numb} loaded successfully!")
        
        # xəbərlərin olduğu siyahını tapır
        soup = BeautifulSoup(response.content, "html.parser")
        news_list = soup.find('ul', class_='news-list')
        
        # hər bir xəbər blokunu tapır
        items = news_list.find_all('li')
        for item in items:
            try:
                # hər bir xəbərin yerləşdiyi ünvan addressini tapır
                link = item.find('a')
                
                # xəbər başlıqlarını götürür
                title = link.get('title').strip()
                
                url_address = base_url + link.get('href', '').strip()
                
                # hər xəbərin daxilinə girib mətnini oxuyur
                news_soup = BeautifulSoup(requests.get(url_address, headers=headers).content, 
                                          "html.parser")
                content = ""
                
                for p in news_soup.select('#NewsContent p'):
                    content += p.text.strip() + " "
                
                # tarix və xəbər baxışlarının qeyd olunduğu hissə
                tools_div = item.select_one('div.tools')
                time_tag = tools_div.find('time')
                # tarixi götürür
                published_date = time_tag.text.strip()
                
                views_div = tools_div.find_all('div')[-1]
                # baxış sayını göstərir
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

# scrape edilən məlumatları dataframe-ə çevirir
df = pd.DataFrame(data=news_data)
df.to_csv('egov_news.csv', encoding='utf-8')

