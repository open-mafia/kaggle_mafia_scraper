
import os
import pandas as pd 

from bay12_scraper.subforum import SubforumAnalyzer
from bay12_scraper.thread import ForumThread 

from textwrap import dedent
import logging
logger = logging.getLogger(__name__)


from prompt_toolkit import (
    print_formatted_text as print,
    HTML, prompt, PromptSession, 
)
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.shortcuts import clear
from prompt_toolkit.styles import Style 
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.validation import Validator, ValidationError


def prompt_in(message, labels=[], force=True, **kwargs):
    if (len(labels) == 0) or not force:
        val = lambda x: True
    else:
        val = lambda x: x in labels
    result = prompt(
        message, 
        completer=WordCompleter(labels), 
        validator=Validator.from_callable(val),
        **kwargs
    )
    return result


# Set up web browser
import webbrowser
firefox_path = r"C:\Program Files\Mozilla Firefox\firefox.exe"
webbrowser.register('firefox', None, webbrowser.BackgroundBrowser(firefox_path), 1)
firefox = webbrowser.get('firefox')

def open_url(url):
    firefox.open_new_tab(url)

#
SKIP = 'skip [!]'

class ThreadLabeler(SubforumAnalyzer):

    LABELS_THREAD = [
        'beginners-mafia', 
        'vanilla', 
        'classic',
        'closed-setup', 
        'byor', 
        'bastard', 
        'vengeful',  
        'paranormal', 
        'cybrid', 
        'supernatural', 
        'kotm', 
        'non-mafia-game', 
        'other', 
        SKIP, 
    ]

    DEFAULT_URL = "http://www.bay12forums.com/smf/index.php?board=20.0"

    def __init__(self, url=None):
        super().__init__(url or self.DEFAULT_URL)
        
    def menu_threads(self, filename=None):
        """Main loop."""

        # Find ones we already know about
        known = []
        if filename:
            try:
                files = pd.read_csv(filename, header=0, encoding='utf-8') 
                known.extend(files['url'])
                print("Found %s existing labels." % len(known))
            except FileNotFoundError:
                pass
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Prepare labels (in a loop)
        labels = []
        for thread_dict in self.threads:
            if thread_dict['url'] in known:
                continue  # we don't need to re-evaluate
            
            try:
                clear()
                srs = self._menu_thread_single(**thread_dict)
                labels.append(srs)

                if filename:  # save immediately so we don't lose work... 
                    df = pd.DataFrame(srs).T
                    with open(filename, 'a', encoding='utf-8') as f:
                        is_first = (f.tell()==0)
                        df.to_csv(f, header=is_first, index=False)

            except Exception:
                logger.exception("Error while parsing %s" % thread_dict['url'])
            except:
                logger.exception("Error while parsing %s" % thread_dict['url'])
                raise

        self.df_labels = pd.DataFrame(labels)
        
    def _menu_thread_single(self, url, name, replies=0):
        """Menu for a thread."""

        # Let user know 
        txt = dedent("""
            "{name}"
            [ {replies} replies ]
            <blue>{url}</blue>
            """.format(url=url, name=name, replies=replies)
        )
        try:
            print(HTML(txt))
        except Exception:
            print(txt)

        # Open browser
        open_url(url)

        # Parse the thread (useful later, maybe not now)
        # thread = ForumThread(url)
        # thread.users
        # thread.posts

        label = prompt_in("Thread type: ", self.LABELS_THREAD)
        if label == SKIP:
            raise ValueError("User is skipping.")

        # Save it
        res = pd.Series(
            [url, name, label, replies], 
            index=['url', 'name', 'label', 'replies'],
        )
        # TODO: Maybe also add more info from a parsed thread? ... maybe later 
        return res


class RoleLabeler(SubforumAnalyzer):
    LABELS_ROLE = [
        'mafia', 'town', 'serial-killer', 'survivor', 
        'cult', 'other-third-party', 
        'game-master', 'ic', 'observer', 'unknown', 
    ]

    # TODO: Implement

