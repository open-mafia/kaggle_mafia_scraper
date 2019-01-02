

import logging 
from ast import literal_eval

import pandas as pd 

from bay12_scraper.thread import ForumThread


def load_or_create_posts(fname, roles=None, threads=None):
    """Loads the Posts dataframe.
    
    If doesn't exist, creates one from roles, threads, and online material.
    
    Parameters
    ----------
    fname : str
        File name to read to/write from.
    roles, threads : pd.DataFrame
        Roles and Threads dataframe.
    """

    try:
        posts = pd.read_csv(fname, header=0, encoding='utf-8')
        # turn string into list of strings
        posts['quotes'] = posts['quotes'].apply(literal_eval)
        return posts
    except FileNotFoundError:
        if not (roles and threads):
            raise
    
    dfs = []
    for t_num in sorted(roles.thread_num.unique()):
        try:
            ft = ForumThread(threads[threads.thread_num == t_num].url.iloc[0])
            df = ft.df.copy()
            df.insert(0, 'thread_num', t_num)
            dfs.append(df)
        except Exception:
            logging.exception('Failure during thread #%s' % t_num)
    posts = pd.concat(dfs, axis='rows')
    try:
        posts.to_csv(fname, index=False, header=True, encoding='utf-8')
    except Exception:
        logging.exception('Could not write to CSV file. Check dir exists?')
    return posts
