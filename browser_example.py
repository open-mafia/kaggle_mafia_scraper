

url = "http://google.com"
url = "http://www.bay12forums.com/smf/index.php?board=20.0"
url = "http://www.bay12forums.com/smf/index.php?topic=48870.0"

import webbrowser


firefox_path = r"C:\Program Files\Mozilla Firefox\firefox.exe"
webbrowser.register('firefox', None, webbrowser.BackgroundBrowser(firefox_path), 1)
firefox = webbrowser.get('firefox')
firefox.open_new_tab(url)


# Alternative we could use Selenium Webdriver
# See: https://automatetheboringstuff.com/chapter11/

#from selenium import webdriver
#browser = webdriver.Firefox()
#browser.get(url)
