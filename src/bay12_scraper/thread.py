"""This module contains a full forum thread parser."""

try: 
    from BeautifulSoup import BeautifulSoup
except ImportError:
    from bs4 import BeautifulSoup

import requests
import re  # regex
import logging
logger = logging.getLogger(__name__)

from collections import Counter

from bay12_scraper.post import parse_forum_page_to_posts  # ForumPost

import pandas as pd 
from prompt_toolkit.shortcuts import ProgressBar

import time 


# url = "http://www.bay12forums.com/smf/index.php?topic=42347.0"
# Later post:
# http://www.bay12forums.com/smf/index.php?topic=42347.350
# Thus we can infer the schema from that

# response = requests.get(url) 
# parsed_html = BeautifulSoup(response.text, 'html.parser') 

# Something to look at later maybe?...
# response = requests.get(url + "&action=.xml") 
# parsed_xml = BeautifulSoup(response.text, 'lxml') 


# Find thread page count
def get_thread_page_count(parsed_html):
    """Returns the number of pages in current thread."""
    page_html = (
        parsed_html
        .find("div", {"id": "postbuttons"})
        .find("div", {"class": "margintop middletext floatleft"})
    )
    nav_pages = page_html.find_all("a", {"class": "navPages"})
    num_pages = 1  # default to 1 page, because... duh.
    for p in nav_pages:
        x = int(p.text)
        if (x > num_pages):
            num_pages = x
    return num_pages


def get_base_url(url):
    """Finds the "base url" to concat with post numbers."""
    return re.compile(r".*topic=[0-9]+\.").findall(url)[0]


def get_topic_num(url):
    """Finds the topic number from the url."""
    tn_reg = r"\?.*topic=[0-9]+\."
    tn_s = r'topic='
    topic_num_raw = re.compile(tn_reg).findall(url)[0]
    topic_num = int(
        topic_num_raw[topic_num_raw.find(tn_s) + len(tn_s):-1]
    )
    return topic_num


# Below is the actual forum thread class


class ForumThread(object):
    """Scrapes a full thread based on any url of it.
    
    Attributes
    ----------
    url : str
    posts : list
        List of all ForumPost's.
    users : list
        List of all usernames that posted.
    name : str
        Name of the topic. Currently, uses final name.
    base_url : str
    topic_num : int
    num_pages : int
    soup : BeautifulSoup
        Parsed given page (from url), not initial page.
    """

    def __init__(self, url, timeout=None, debug_print=False):
        self.url = url
        self.base_url = get_base_url(url)
        self.topic_num = get_topic_num(url)
        self.timeout = timeout

        response = requests.get(url, timeout=self.timeout)  
        # TODO: maybe change to base_url?
        self.soup = BeautifulSoup(response.text, 'lxml')  # html.parser') ?
        self.num_pages = get_thread_page_count(self.soup)
        self.sub_urls = [
            # posts are split into groups of 15 by SMF
            self.base_url + str(i * 15)
            for i in range(self.num_pages)
        ]

        # Get individual posts by reading pages
        self.posts = []
        # NOTE: Removing the ProgressBar temporarily, because it just liked 
        # to mess up, sadly... TODO: Report bug? Maybe 
        #
        # if True:
        with ProgressBar(title="#{}".format(self.topic_num)) as pb:
            if debug_print:
                print("#%s (%s pages)" % (self.topic_num, len(self.sub_urls))) 

            # for sub_url in self.sub_urls:
            for sub_url in pb(self.sub_urls):
                retry = True
                while retry:
                    try:
                        logger.info('Parsing url: %s' % sub_url)
                        page_posts = parse_forum_page_to_posts(
                            sub_url, timeout=self.timeout
                        )
                        self.posts.extend(page_posts)
                        retry = False
                    except Exception:
                        logger.exception(
                            "Error while parsing url: %s" % sub_url)
                        if debug_print:
                            print("Error while parsing url: %s" % sub_url)
                        time.sleep(self.timeout)

        # Filter out users 
        
        self.users = Counter()
        for p in self.posts:
            self.users[p.user] += 1

        # Old version
        # self.users = list(OrderedDict.fromkeys((p.user for p in self.posts)))

        # Get thread name
        # TODO: Maybe get original name instead of final name?
        # Then would need to parse the first reply to the first post... 
        # Currently we use the easiest one though.
        sp_topic = self.soup.find('span', {'id': 'top_subject'})
        self.name = sp_topic.text[
            len('Topic: '):
            sp_topic.text.find('\xa0') - 1
        ]

        # TODO: Calc other meta-info?
        # nah

    def __repr__(self):
        return "[ForumThread #%s with %s users and %s posts]: %r" % (
            self.topic_num, len(self.users), len(self.posts), 
            self.name, 
        )

    @property
    def df(self):
        """Returns a pandas DataFrame representing self."""
        if hasattr(self, '_df'):
            return self._df
        all_srs = [pp.to_series() for pp in self.posts]
        self._df = res = pd.DataFrame(all_srs)
        return res


# t = ForumThread("http://www.bay12forums.com/smf/index.php?topic=42347.0")  
# t = ForumThread("http://www.bay12forums.com/smf/index.php?topic=42347.360")

