
try: 
    from BeautifulSoup import BeautifulSoup
except ImportError:
    from bs4 import BeautifulSoup

from copy import copy 
import requests
import logging 


QUOTE_REPLACEMENT = " QUOTED_SECTION "


class Post(object):

    def __init__(self, user, text, quotes=[], **kwargs):
        self.user = user 
        self.text = text 
        self.quotes = quotes 
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return "Post(%r, %s symbols, %s quotes)" % (
            self.user, len(self.text), len(self.quotes), 
        )

    @classmethod
    def from_soup(cls, soup):

        user = (
            soup
            .find("div", {'class': 'poster'})
            .find("a")
            .text
        )

        wrapped = (
            soup
            .find("div",{'class':'post'})
            .find("div",{'class':'inner'})
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
                quotes.append(qt_body) 
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


def parse_forum_page_to_posts(url):
    """Finds all posts from a url, parses to Post."""

    response = requests.get(url) 
    soup = BeautifulSoup(response.text, "html.parser")  # html5lib?

    wrapped_posts = (
        soup
        .find("div",{"id": "forumposts"})
        .find("form")
        .children
    )
    posts = []
    for wp in wrapped_posts:
        try:
            posts.append(Post.from_soup(wp))
        except Exception:
            # logging.exception('Error parsing post.')
            pass  # logging is too annoying here :P 
    return posts
