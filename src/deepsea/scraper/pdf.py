
from os import PathLike, listdir, path

import pdfplumber

from nltk import FreqDist
from nltk.corpus import names, stopwords
from nltk.tokenize import word_tokenize

from dataclasses import dataclass


@dataclass(slots=True)
class Person:
    name:str
    address:str
    email:str


class PdfScraper:

    def __init__(self):
        pass

    def scrape(self):
        print("scrape it!!!!")

