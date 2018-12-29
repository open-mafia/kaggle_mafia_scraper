"""Gets all thread info from a subforum."""

try: 
    from BeautifulSoup import BeautifulSoup
except ImportError:
    from bs4 import BeautifulSoup

import requests
import re  # regex

from prompt_toolkit.shortcuts import ProgressBar

import logging
logger = logging.getLogger(__name__)


class SubforumAnalyzer(object):
    """Gets all post info from a subforum link."""

    def __init__(self, url):
        # Analyze the initial page
        self.url = url 
        self.base_url = base_url = (
            re.compile(r".*board=[0-9]+\.").findall(url)[0]
        )

        i_response = requests.get(base_url + '0')
        i_soup = BeautifulSoup(i_response.text, 'html.parser')

        n_pages_total = int(
            i_soup
            .find('div', {'id': 'modbuttons_bottom'})
            .find_all('a', {'class': 'navPages'})[-1]
            .text
        )

        # Go through each page and get all thread info
        self.threads = []
        with ProgressBar() as pb:
            for i in pb(range(n_pages_total), total=n_pages_total):
                s_response = requests.get(base_url + str(20*i))
                s_soup = BeautifulSoup(s_response.text, 'html.parser')

                # Find table of threads
                tbl = s_soup.find('div', {'id': 'messageindex'}).find('tbody')
                rows = tbl.find_all('tr')

                # Parse all rows
                for row in rows:
                    link = row.find('td', {'class': 'subject'}).find('a')
                    thread = dict(
                        url = link['href'], 
                        name = link.text, 
                        replies = int(
                            row.find('td', {'class': 'replies'}).text.strip()
                        ), 
                    )
                    self.threads.append(thread)

        # We now have all thread info, and are done.


# sf = SubforumAnalyzer("http://www.bay12forums.com/smf/index.php?board=20.0")
