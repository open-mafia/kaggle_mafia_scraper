
import pandas as pd 
from bay12_scraper.menu import ThreadLabeler 

OUTPUT_THREADS = 'output/threads.csv'

if __name__ == "__main__":
    TL = ThreadLabeler()
    TL.menu_threads(filename=OUTPUT_THREADS)
