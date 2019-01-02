

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


def fix_roles_df(df_roles):
    df_roles_fixed = df_roles.copy()

    tbr = (df_roles_fixed['role'] == 'replaced')
    df_roles_fixed['replacement_depth'] = 0

    cols_fixed = list(df_roles.columns) + ['replacement_depth']

    while tbr.sum():
        df_roles_fixed = df_roles_fixed.merge(
            df_roles_fixed, how='left', 
            left_on=['thread_num', 'replaced_by'], 
            right_on=['thread_num', 'user'], 
            suffixes=('', '_repl')
        )[cols_fixed + ['role_repl']]
        
        tbr = (df_roles_fixed['role'] == 'replaced')
        df_roles_fixed['role'][tbr] = df_roles_fixed['role_repl'][tbr]
        df_roles_fixed['replacement_depth'][tbr] += 1
        df_roles_fixed = df_roles_fixed[cols_fixed]
    
    return df_roles_fixed


def split_ds(roles, posts, pct_public=0.2, pct_private=0.2):
    """Splits the dataset into 3 parts: train, test (public), test (private).
    
    Returns dict of [posts, roles] for keys ['train', 'public', 'private'].
    """

    pct_train = 1 - pct_public - pct_private
    assert min(pct_train, pct_public, pct_private) > 0
    
    t_nums = sorted(roles['thread_num'].unique())
    
    # Very simple time-based split, since thread numbers are chronological 
    # I know this introduces a bias - will probably change later
    
    n_prv = int(pct_private * len(t_nums))
    n_pub = int(pct_public * len(t_nums))
    
    t_prv = t_nums[-n_prv:]
    t_pub = t_nums[-(n_prv + n_pub): -n_prv]
    # NOTE: The remainder will round up, which is OK 
    
    # Split the posts
    pi_prv = posts.thread_num.isin(t_prv)
    pi_pub = posts.thread_num.isin(t_pub)
    
    p_private = posts[pi_prv]
    p_public = posts[pi_pub]
    p_train = posts[~(pi_pub | pi_prv)]
    
    # Split the roles
    ri_prv = roles.thread_num.isin(t_prv)
    ri_pub = roles.thread_num.isin(t_pub)
    
    r_private = roles[ri_prv]
    r_public = roles[ri_pub]
    r_train = roles[~(ri_pub | ri_prv)]
    
    # Return them D:
    
    res = {
        'train': [p_train, r_train], 
        'public': [p_public, r_public], 
        'private': [p_private, r_private]
    }
    return res
