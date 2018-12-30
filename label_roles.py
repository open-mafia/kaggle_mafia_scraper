
import pandas as pd 
from bay12_scraper.menu import RoleLabeler 

if __name__ == "__main__":
    TL = RoleLabeler(folder='./output') # 
    TL.menu_roles()
