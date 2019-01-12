

# import logging 
import pandas as pd 

from bay12_scraper.thread import ForumThread
from bay12_scraper.prep import quote_str_to_list


def load_or_create_extended_posts(
    fname, threads, incremental=True, timeout=10, limit=None, 
):
    """Loads the Extended Posts dataframe.
    
    If doesn't exist, creates one from threads and online material.
    Works incrementally
    
    Parameters
    ----------
    fname : str
        File name to read to/write from.
    threads : pd.DataFrame
        Threads definition dataframe.
    """

    threads = threads.iloc[:limit]
    
    res_needs_header = False
    try:
        eposts = pd.read_csv(fname, header=0, encoding='utf-8')
        # turn string into list of strings
        eposts['quotes'] = eposts['quotes'].apply(quote_str_to_list)
        # res_needs_header = False  # set above
    except FileNotFoundError:
        eposts = pd.DataFrame(columns=['thread_num', 'user', 'text', 'quotes'])
        res_needs_header = True

    print('TOTAL THREADS: %s' % len(threads))

    for i in range(len(threads)):
        t = threads.iloc[i]
        # url, thread_num, thread_name, thread_label, thread_replies
        if t.thread_num in eposts.thread_num.values:
            continue
        
        ft = ForumThread(t.url, timeout=timeout)
        df = ft.df.copy()
        df.insert(0, 'thread_num', t.thread_num)
        if incremental:
            df.to_csv(
                fname, mode='a', index=False, header=res_needs_header, 
                encoding='utf-8'
            )
        eposts = eposts.append(df)
        res_needs_header = False

        if ((i + 1) % 20 == 0):
            print('Saved threads: %s' % i)

    if not incremental:
        eposts.to_csv(
            fname, index=False, header=True, 
            encoding='utf-8'
        )        
    return eposts


def split_extended_ds(threads, eposts, pct_public=0.2, pct_private=0.2):
    """Splits the dataset into 3 parts: train, test (public), test (private).
    
    Returns dict of [posts, threads] for keys ['train', 'public', 'private'].
    """

    pct_train = 1 - pct_public - pct_private
    assert min(pct_train, pct_public, pct_private) > 0
    
    t_nums = sorted(threads['thread_num'].unique())

    # Very simple time-based split, since thread numbers are chronological 
    # I know this introduces a bias - will probably change later
    
    n_prv = int(pct_private * len(t_nums))
    n_pub = int(pct_public * len(t_nums))
    
    t_prv = t_nums[-n_prv:]
    t_pub = t_nums[-(n_prv + n_pub): -n_prv]
    # NOTE: The remainder will round up, which is OK 

    # Split the posts
    pi_prv = eposts.thread_num.isin(t_prv)
    pi_pub = eposts.thread_num.isin(t_pub)
    
    p_private = eposts[pi_prv]
    p_public = eposts[pi_pub]
    p_train = eposts[~(pi_pub | pi_prv)]

    # Split the threads (with labels)
    ri_prv = threads.thread_num.isin(t_prv)
    ri_pub = threads.thread_num.isin(t_pub)
    
    r_private = threads[ri_prv]
    r_public = threads[ri_pub]
    r_train = threads[~(ri_pub | ri_prv)]

    res = {
        'train': [p_train, r_train], 
        'public': [p_public, r_public], 
        'private': [p_private, r_private]
    }
    return res
