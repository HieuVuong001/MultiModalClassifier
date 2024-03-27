from bs4 import BeautifulSoup
import os
import requests
import time
headers = {
    "Connection": "keep-alive",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36"
}


base_url = 'https://pokemon.fandom.com/wiki'

# site = requests.get(base_url, headers=headers)
# soup = BeautifulSoup(site.text, 'html.parser')

# with open('sample.txt', 'w') as f:
#     f.write(site.text)


types = [ 'Fire', 'Water', 'Electric', 'Grass', 'Ice', 'Fighting', 'Normal']


for pokemon_type in types:
    url = f'{base_url}/{pokemon_type}_type'
    
    site = requests.get(url, headers=headers)

    soup = BeautifulSoup(site.text, 'html.parser')

    items = soup.find_all('div', class_='wikia-gallery-item')
    
    try:
        os.makedirs(f'./pokemon/{pokemon_type}')
    except OSError as error:
        print(error)
    
    for item in items:

        download_url = item.div.div.a.img['src']
        title = item.div.div.a.img['title']
        # print(f'{base}{ext}')
        downloaded_img = requests.get(download_url).content
        print(f'Downloading {title}')
        with open(f'./pokemon/{pokemon_type}/{title}.png', 'wb') as f:
            f.write(downloaded_img)
        time.sleep(0.01)

