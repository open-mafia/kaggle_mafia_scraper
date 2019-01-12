

import logging 
from ast import literal_eval

import pandas as pd 

from bay12_scraper.thread import ForumThread


def load_or_create_posts(fname, roles=None, threads=None, incremental=False):
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
        if (roles is None) or (threads is None):
            raise
    
    dfs = []
    i = 0
    for t_num in sorted(roles.thread_num.unique()):
        try:
            i += 1
            ft = ForumThread(threads[threads.thread_num == t_num].url.iloc[0])
            df = ft.df.copy()
            df.insert(0, 'thread_num', t_num)
            dfs.append(df)
        except Exception:
            logging.exception('Failure during thread #%s' % t_num)
            continue
        
        if incremental:
            df.to_csv(
                fname, mode='a', index=False, header=(i == 0), 
                encoding='utf-8'
            )

    posts = pd.concat(dfs, axis='rows')
    if not incremental:
        posts.to_csv(
            fname, index=False, header=(i == 0), 
            encoding='utf-8'
        )
    return posts


def load_or_create_extended_posts(fname, threads=None, incremental=True):
    """Loads the EPosts dataframe.
    
    If doesn't exist, creates one from threads and online material.
    
    Parameters
    ----------
    fname : str
        File name to read to/write from.
    roles, threads : pd.DataFrame
        Roles and Threads dataframe.
    """
    
    try:
        eposts = pd.read_csv(fname, header=0, encoding='utf-8')
        # turn string into list of strings
        eposts['quotes'] = eposts['quotes'].apply(literal_eval)
        return eposts
    except FileNotFoundError:
        if (threads is None):
            raise

    print('TOTAL THREADS: %s' % len(threads))

    dfs = []
    for i, row in threads.iterrows():
        try:
            t_url = row.url
            t_num = row.thread_num 
            ft = ForumThread(t_url)
            df = ft.df.copy()
            df.insert(0, 'thread_num', t_num)
            dfs.append(df)

            if ((i + 1) % 20 == 0):
                print('Saved threads: %s' % i)
        except Exception:
            logging.exception('Failure during thread #%s' % t_num)
            continue
        
        if incremental:
            df.to_csv(
                fname, mode='a', index=False, header=(i == 0), 
                encoding='utf-8'
            )        
    eposts = pd.concat(dfs, axis='rows')
    if not incremental:
        eposts.to_csv(
            fname, index=False, header=(i == 0), 
            encoding='utf-8'
        )        
    return eposts


def fix_roles_df(df_roles):
    """Fixes quirks with replacement. 
    
    Adds replacement_depth and final_player columns, while 
    also assigning the appropriate roles to each player.
    """

    df_fixed = df_roles.copy()

    df_fixed['replacement_depth'] = 0
    df_fixed['final_player'] = float('nan')

    cols_fixed = (
        list(df_roles.columns) + 
        ['replacement_depth', 'final_player']
    )

    # tbr == "to be replaced"
    tbr = (df_fixed['role'] == 'replaced')
    # Set "original player" roles
    df_fixed['final_player'][~tbr] = df_fixed['user'][~tbr]

    # Loop while we still have folks we need to replace
    while tbr.sum():
        df_fixed = df_fixed.merge(
            df_fixed, how='left', 
            left_on=['thread_num', 'replaced_by'], 
            right_on=['thread_num', 'user'], 
            suffixes=('', '_r')
        )[cols_fixed + ['role_r', 'final_player_r']]
        
        # Find who still needs to be replaced
        tbr = (df_fixed['role'] == 'replaced')

        # Replace the right-hand role
        df_fixed['role'][tbr] = df_fixed['role_r'][tbr]
        # Replace the right-hand 'final player'
        df_fixed['final_player'][tbr] = df_fixed['final_player_r'][tbr]
        # Add one to replacement depth
        df_fixed['replacement_depth'][tbr] += 1
        df_fixed = df_fixed[cols_fixed]
    
    return df_fixed


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
