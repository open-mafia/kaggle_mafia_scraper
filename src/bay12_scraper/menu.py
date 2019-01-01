
import os
import pandas as pd 

from bay12_scraper.subforum import SubforumAnalyzer
from bay12_scraper.thread import ForumThread, get_topic_num 

from textwrap import dedent
import logging
logger = logging.getLogger(__name__)

import warnings

from prompt_toolkit import (
    print_formatted_text as print,
    HTML, prompt, 
)
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.validation import Validator


def prompt_in(message, labels=[], force=True, **kwargs):
    if (len(labels) == 0) or not force:
        def val(x):
            return True
    else:
        def val(x):
            return x in labels
    result = prompt(
        message, 
        completer=WordCompleter(labels), 
        validator=Validator.from_callable(val),
        **kwargs
    )
    return result


# Set up web browser
import webbrowser
# ffp = r"C:\Program Files\Mozilla Firefox\firefox.exe"
# webbrowser.register('firefox', webbrowser.BackgroundBrowser(ffp), 1)
# firefox = webbrowser.get('firefox')


def open_url(url):
    webbrowser.open_new_tab(url)


def print_thread(url, name='', replies=0):
    print(HTML(dedent("""
        {header}
        "{name}"
        [ {replies} replies ]
        <blue>{url}</blue>
        """.format(header=SEPARATOR, url=url, name=name, replies=replies)
    )))        


#

SEPARATOR = "============================="

REPLACED = 'replaced'
SKIP = 'skip [!]'


class ThreadLabeler(SubforumAnalyzer):
    """Object for labelling threads."""

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
    DEFAULT_FOLDER = './output'

    def __init__(self, url=None, folder=None):
        self.folder = os.path.abspath(folder or self.DEFAULT_FOLDER)
        os.makedirs(self.folder, exist_ok=True)
        super().__init__(url or self.DEFAULT_URL)
        
    def menu_threads(self, threads_file='threads.csv'):
        """Menu loop for labeling thread types.
        
        Output location: {folder}/{threads_file} (as .csv)
        Columns: [url, thread_num, thread_name, thread_label, thread_replies]
        """

        res_cols = [
            'url', 'thread_num', 'thread_name', 
            'thread_label', 'thead_replies', 
        ]

        filename = os.path.join(self.folder, threads_file)

        # Find ones we already know about
        known_threads = pd.DataFrame(columns=res_cols)
        try:
            known_threads = pd.read_csv(filename, header=0, encoding='utf-8') 
            print("Found %s existing labels." % len(known_threads))
        except FileNotFoundError:
            pass

        # Prepare labels (in a loop)
        labels = []
        for thread_dict in self.threads:
            url = thread_dict['url']
            thread_num = get_topic_num(url)
            name = thread_dict.get('name', '')
            replies = thread_dict.get('replies', 0)
            
            if url in known_threads.url.values:
                continue  # we don't need to re-evaluate
            
            try:
                open_url(url)
                # clear()
                print_thread(**thread_dict)

                # Get label
                label = prompt_in("Thread type: ", self.LABELS_THREAD)
                if label == SKIP:
                    continue

                df = pd.DataFrame(
                    [[url, thread_num, name, label, replies]], 
                    columns=res_cols
                )
                with open(filename, 'a', encoding='utf-8') as f:
                    is_first = (f.tell() == 0)
                    df.to_csv(f, header=is_first, index=False)
                labels.append(df)
            except Exception:
                logger.exception("Error while parsing %s" % thread_dict['url'])
        self.labels_threads = pd.concat([known_threads] + labels, axis=0)
        

class RoleLabeler(object):

    LABELS_ROLE = [
        'unknown', 
        'mafia', 'town', 'serial-killer', 'survivor', 
        'cult', 'other-third-party', 
        'game-master', 'ic', 'observer', 
        REPLACED, 
        SKIP
    ]
    UNKNOWN_ROLE = 'unknown'
    DEFAULT_FOLDER = './output'

    def __init__(self, folder=None):
        self.folder = os.path.abspath(folder or self.DEFAULT_FOLDER)

    def find_thread_df(self, thread_srs, posts_folder='posts'):
        """Loads a thread df from disk, or generates from online posts."""

        url = thread_srs['url']
        num = thread_srs['thread_num']

        os.makedirs(os.path.join(self.folder, posts_folder), exist_ok=True)
        location = os.path.join(self.folder, posts_folder, "%s.csv" % num)
        try:
            res = pd.read_csv(location, header=0, encoding='utf-8')
            return res
        except Exception:
            pass
        
        thread = ForumThread(url)
        res = thread.df
        
        # save it to location before returning
        res.to_csv(location, header=True, index=False, encoding='utf-8')

        return res

    def menu_roles(
        self, relevant_thread_labels=['vanilla', 'beginners-mafia'],
        threads_file='threads.csv', roles_file='roles.csv', 
        posts_folder='posts', 
    ):
        """Main loop for labelling user roles, as well as outputting posts.

        Output locations: {folder}/{roles_file} (as .csv)
        Columns: [url, user, role, num_posts]

        Post output:
            {folder}/{posts_folder}/{thread_num}.csv
        
        """

        res_cols = ['thread_num', 'user', 'role', 'num_posts', 'replaced_by']

        full_roles_file = os.path.join(self.folder, roles_file)
        full_threads_file = os.path.join(self.folder, threads_file)

        # Read relevant threads
        all_threads = pd.read_csv(
            full_threads_file, 
            header=0, encoding='utf-8'
        )
        threads = all_threads[
            all_threads.thread_label.isin(relevant_thread_labels)
        ]
        # with cols: 
        # ['url', 'thread_num', 'thread_name', 'thread_label', 'thead_replies']

        # Load dataframe, list of preexisting parsed roles
        role_df = pd.DataFrame(columns=res_cols)
        complete_roles_nums = []
        try:
            role_df = pd.read_csv(
                full_roles_file, 
                header=0, encoding='utf-8', 
            )
            complete_roles_nums = (
                role_df.groupby('thread_num')
                .filter(lambda x: (x['role'] != self.UNKNOWN_ROLE).all())
            )['thread_num'].tolist()
        except FileNotFoundError:
            pass

        # Prepare labels (in a loop)
        labels = []
        for _, t in threads.iterrows():
            url = t.url
            thread_num = t.thread_num

            if thread_num in complete_roles_nums:
                continue  # we don't need to re-evaluate
            
            try:
                open_url(url)
                # clear()
                print_thread(url, t.thread_name, t.thread_replies)

                # Parse the thread's posts
                thread_df = self.find_thread_df(t, posts_folder=posts_folder)
                # cols are ['user', 'text', 'quotes']

                # Try loading existing
                if thread_num in role_df.thread_num.values:
                    dfu = role_df[role_df.thread_num == thread_num]
                else: 
                    # Create user dataframe
                    gbnt = thread_df.groupby('user').text.count()
                    dfu = pd.DataFrame({
                        'num_posts': gbnt,
                        'url': url,
                        'thread_num': thread_num, 
                        'role': self.UNKNOWN_ROLE,
                        'replaced_by': '',
                    }, index=gbnt.index)
                    # NOTE: index is 'user', because of the groupby

                    # Try to infer users from the initial post
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')

                        op = thread_df.loc[0, 'text']
                        for uname in dfu.index:
                            if (op.lower().find(uname.lower()) == -1):
                                # assume if name not in OP then they 
                                # aren't playing
                                dfu.role[uname] = 'observer'
                        # original poster is usually GM 
                        dfu.role[thread_df.loc[0, 'user']] = 'game-master'  
                        
                while True:  
                    # Exits when SKIP is entered.
                    print(dfu[['num_posts', 'role', 'replaced_by']])
                    print(SKIP)

                    user = prompt_in("User: ", list(dfu.index) + [SKIP])
                    if user == SKIP:
                        break

                    print(self.LABELS_ROLE)
                    label = prompt_in("Role: ", self.LABELS_ROLE)
                    if label == SKIP:
                        continue
                    elif label == REPLACED:
                        repl_by = prompt_in(
                            "Replaced by: ", list(dfu.index) + [SKIP]
                        )
                        if repl_by == SKIP:
                            continue
                        # Replacement
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            dfu.replaced_by[user] = repl_by
                    # Role (or 'replaced')
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        dfu.role[user] = label

                # Save immediately so we don't lose work...
                res = dfu.reset_index()[res_cols]

                with open(full_roles_file, 'a', encoding='utf-8') as f:
                    is_first = (f.tell() == 0)
                    res.to_csv(f, header=is_first, index=False)
                labels.append(res)

            except Exception:
                logger.exception("Error while parsing %s" % url)

        self.labels_roles = pd.concat(labels, axis=0)
