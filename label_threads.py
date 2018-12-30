
import pandas as pd 
from bay12_scraper.menu import ThreadLabeler

if __name__ == "__main__":
    TL = ThreadLabeler(folder='./output') # 
    TL.menu_threads()
