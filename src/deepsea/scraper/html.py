import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass


@dataclass
class HtmlScraper:
    required_content: str
    url: str

    def pour_soup(self) -> BeautifulSoup:
        # get the web page content
        r = requests.get(self.url)

        # Convert to a beautiful Soup object
        soup = BeautifulSoup(r.content)
        print(soup.prettify())
        return soup

    def scrape_soup(self, body_content):
        pass
