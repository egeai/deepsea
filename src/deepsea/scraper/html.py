import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from urllib.parse import urljoin


@dataclass
class HtmlScraper:
    required_content: str
    url: str

    def pour_soup(self) -> BeautifulSoup:
        # get the web page content
        r = requests.get(self.url)
        #print(r.status_code, r.encoding, r.text)
        #https://quotes.toscrape.com/
        base_url = 'https://www.imdb.com'

        # Convert to a beautiful Soup object
        soup = BeautifulSoup(r.text, 'html.parser')
        trows = soup.find('tbody', {'class' : 'lister-list'}).findAll('tr')
        with open('naber.csv', 'w') as f:
            for trow in trows:
                tdata = trow.find('td', {'class':'titleColumn'})
                rating = trow.find('td', {'class':'ratingColumn'})
                url = urljoin(base_url, tdata.a["href"])
                print(tdata.a.string)
                # GO TO SUB PAGE
                r_sub = requests.get(url).text
                soup_sub = BeautifulSoup(r_sub, 'html.parser')
                div = soup_sub.find('div', {'class':'iJtmbR'})
                presentation_li_s = div.find('ul', {'class':'ipc-inline-list'}).findAll('li', {'role':'presentation'})
                print("length is ", len(presentation_li_s))
                for presentation_li in presentation_li_s:
                    if presentation_li.span:
                        print(presentation_li.span.string)
                    else:
                        print(presentation_li.text)
                print("END **********************")

                # print(tdata.a.string, tdata.span.string, rating.strong.string)
                # print(tdata.a["href"])
                # f.write('The Movie is {}, {}, {}'.format(tdata.a.string, tdata.span.string, rating.strong.string))
                # f.write('\n')

        # print(soup.findAll('td', {'class':'titleColumn'}))
        # for tag in soup.findAll('span', {'class': 'text'}):
        """
        with open('naber.csv', 'w') as f:
            for tag in soup.findAll('small', {'class': 'author'}):
                f.write(tag.string)
                f.write('\n')
        """
        return soup

    def scrape_soup(self, body_content):
        pass
