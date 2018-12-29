
import os
import pandas as pd 

from bay12_scraper.subforum import SubforumAnalyzer
from bay12_scraper.thread import ForumThread 

from textwrap import dedent


from prompt_toolkit import (
    print_formatted_text as print,
    HTML, prompt, PromptSession, 
)
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.shortcuts import clear
from prompt_toolkit.styles import Style 
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.validation import Validator, ValidationError


def prompt_in(message, labels, **kwargs):
    result = prompt(
        message, 
        completer=WordCompleter(labels), 
        validator=Validator.from_callable(lambda x: x in labels),
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


class ThreadLabeler(SubforumAnalyzer):

    LABELS_THREAD = [
        'vanilla', 'classic', 'byor', 'closed-setup', 'bastard', 'non-game', 
    ]

    DEFAULT_URL = "http://www.bay12forums.com/smf/index.php?board=20.0"

    def __init__(self, url=None):
        super().__init__(url or self.DEFAULT_URL)
        
    def menu_threads(self):
        """Main loop."""

        labels = []

        for thread_dict in self.threads:
            clear()
            srs = self._menu_thread_single(**thread_dict)
            labels.append(srs)

        self.df_labels = pd.DataFrame(labels)
        
    def _menu_thread_single(self, url, name, replies=''):
        """Menu for a thread."""

        # Let user know 
        print(HTML(dedent("""
            "{name}"
            [ {replies} replies ]
            <blue>{url}</blue>
            """.format(url=url, name=name, replies=replies)
        )))

        # Open browser
        open_url(url)

        # Parse the thread (useful later, maybe not now)
        # thread = ForumThread(url)
        # thread.users
        # thread.posts

        label = prompt_in("Thread type: ", self.LABELS_THREAD)
        
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

