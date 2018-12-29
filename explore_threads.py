
import pandas as pd 

OUTPUT_THREADS = 'output/threads.csv'
raw = pd.read_csv(OUTPUT_THREADS, encoding='utf-8')

threads = (
    raw.groupby('label').url.count()
    .rename('threads').sort_values(ascending=False)
)
replies = (
    raw.groupby('label').replies.sum()
    .rename('replies').sort_values(ascending=False)
)

stats = pd.DataFrame([threads, replies]).T

# set usefulness :P
stats['usefulness'] = 0
stats.usefulness[['beginners-mafia', 'vanilla']] = 3
stats.usefulness[['classic', 'vengeful', 'supernatural']] = 2
stats.usefulness[['closed-setup', 'paranormal', 'kotm']] = 1
stats.usefulness[['byor', 'bastard', 'cybrid']] = 1

print(stats)
