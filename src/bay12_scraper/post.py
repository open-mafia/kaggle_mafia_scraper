"""This module contains the ForumPost and a page parser to get posts."""

try: 
    from BeautifulSoup import BeautifulSoup
except ImportError:
    from bs4 import BeautifulSoup

from copy import copy 
import requests
import time 


QUOTE_REPLACEMENT = " QUOTED_SECTION "


class ForumPost(object):
    """A single forum post, with omitted quotes.

    Attributes
    ----------
    user : str
        The poster's username.
    text : str
        The post's text, with quotes substitued for QUOTE_REPLACEMENT.
    quotes : list
        The quotes from the submitted text.
    raw_soup : BeautifulSoup
        The soup this post was parsed from.

    TODO
    ----
    Try to get the timestamp, maybe? At worst we just have the 
    submission order, which isn't bad at all. 
    """

    def __init__(self, user, text, quotes=[], raw_soup=None, **kwargs):
        self.user = user 
        self.text = text 
        self.quotes = quotes 
        self.raw_soup = raw_soup
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_series(self):
        """Returns a pandas Series representing self."""
        import pandas as pd
        
        res = pd.Series({
            'user': self.user, 
            'text': self.text, 
            'quotes': self.quotes, 
        }).loc[['user', 'text', 'quotes']] 
        return res

    def __repr__(self):
        return "ForumPost(%r, %s symbols, %s quotes)" % (
            self.user, len(self.text), len(self.quotes), 
        )

    @classmethod
    def from_soup(cls, soup):
        """Parses the ForumPost from some soup. 

        See parse_forum_page_to_posts for how to prepare it.
        """
        user = (
            soup
            .find("div", {'class': 'poster'})
            .find("a")
            .text
        )

        wrapped = (
            soup
            .find("div", {'class': 'post'})
            .find("div", {'class': 'inner'})
        )

        # Save old wrapped soup
        dirty = copy(wrapped)

        # Find and replace all quotes, while saving their text.
        quotes = []  # (header, body)
        qt_header = qt_body = qt_footer = True
        while (qt_header or qt_body or qt_footer):
            qt_header = wrapped.find("div", {"class": "quoteheader"})
            if qt_header:
                qt_header.extract()  # or .replace_with("")

            qt_body = wrapped.find("blockquote")
            if qt_body:
                qt_txt_repr = str(qt_body)
                quotes.append(qt_txt_repr) 
                qt_body.replace_with(QUOTE_REPLACEMENT)

            qt_footer = wrapped.find("div", {"class": "quotefooter"})
            if qt_footer:
                qt_footer.extract()

        # Clean out <br/> tags by swapping with /n
        for br in wrapped.find_all('br'):
            br.replace_with('\n')

        # TODO: think - need more cleaning?

        # We now have a clean copy
        cleaned = wrapped.text

        # Create the post
        return cls(user=user, text=cleaned, quotes=quotes, raw_soup=dirty)


def parse_forum_page_to_posts(url, timeout=None):
    """Finds all posts from a url, parses to ForumPost."""

    try:
        response = requests.get(url, timeout=timeout) 
    except requests.Timeout:
        time.sleep(timeout * 2)
        return parse_forum_page_to_posts(url, timeout=timeout * 1.5)

    soup = BeautifulSoup(response.text, "html.parser")  # html5lib?

    wrapped_posts = (
        soup
        .find("div", {"id": "forumposts"})
        .find("form")
        .children
    )
    posts = []
    for wp in wrapped_posts:
        try:
            posts.append(ForumPost.from_soup(wp))
        except Exception:
            # logging.exception('Error parsing post.')
            pass  # logging is too annoying here :P 
    return posts
