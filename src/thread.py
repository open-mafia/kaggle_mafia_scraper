
try: 
    from BeautifulSoup import BeautifulSoup
except ImportError:
    from bs4 import BeautifulSoup

import requests
import re  # regex
import logging
logger = logging.getLogger(__name__)

from collections import OrderedDict

from post import Post, parse_forum_page_to_posts


#url = "http://www.bay12forums.com/smf/index.php?topic=42347.0"
# Later post:
# http://www.bay12forums.com/smf/index.php?topic=42347.350
# Thus we can infer the schema from that

#response = requests.get(url) 
#parsed_html = BeautifulSoup(response.text, 'html.parser') 

# Something to look at later maybe?...
# response = requests.get(url + "&action=.xml") 
# parsed_xml = BeautifulSoup(response.text, 'lxml') 


# Find thread page count
def get_thread_page_count(parsed_html):
    """Returns the number of pages in current thread."""
    page_html = (
        parsed_html
        .find("div",{"id":"postbuttons"})
        .find("div",{"class":"margintop middletext floatleft"})
    )
    nav_pages = page_html.find_all("a",{"class":"navPages"})
    num_pages = 0
    for p in nav_pages:
        x = int(p.text)
        if (x>num_pages):
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
        topic_num_raw[topic_num_raw.find(tn_s)+len(tn_s):-1]
    )
    return topic_num


# Below is the actual forum thread class


class ForumThread(object):
    """Extracts info from """

    def __init__(self, url):
        self.url = url
        self.base_url = get_base_url(url)
        self.topic_num = get_topic_num(url)
        response = requests.get(url) 
        self.soup = BeautifulSoup(response.text, 'html.parser')  # 'html5lib' ?
        self.num_pages = get_thread_page_count(self.soup)
        self.sub_urls = [
            # posts are split into groups of 15 by SMF
            self.base_url + str(i * 15)
            for i in range(self.num_pages)
        ]

        # Get individual posts by reading pages
        self.posts = []
        for sub_url in self.sub_urls:
            logger.info('Parsing url: %s' % sub_url)
            page_posts = parse_forum_page_to_posts(sub_url)
            self.posts.extend(page_posts)

        # Filter out users 
        
        self.users = list(OrderedDict.fromkeys((p.user for p in self.posts)))

        # TODO: Get thread name
        # TODO: Calc other meta-info?

    def __repr__(self):
        return "ForumThread(topic_num=%s, %s users, %s posts)" % (
            self.topic_num, len(self.users), len(self.posts), 
        )


# t = ForumThread("http://www.bay12forums.com/smf/index.php?topic=42347.0")  
# t = ForumThread("http://www.bay12forums.com/smf/index.php?topic=42347.360")

