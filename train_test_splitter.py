import pandas as pd
from collections import Counter
from data_reader import GetDataAsPython

storage_directory = '.'
data = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_repo_specific_final.json")
# data_eslint = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_eslint_final.json") # TODO: should I use this?
repos = Counter([item.repo for item in data])
sorted_repos = sorted(repos.items(), key=lambda d: d[1], reverse=True)
split_point = 300
test_repos = sorted_repos[:split_point:2]
train_repos = sorted_repos[1:split_point:2] + sorted_repos[split_point:] #TODO: improve it so that we have enough test for each warning
train_df = pd.DataFrame(train_repos, columns=['repo', 'samples'])
train_df['train'] = True
test_df = pd.DataFrame(test_repos, columns=['repo', 'samples'])
test_df['train'] = False
repos_df = pd.concat([train_df, test_df], ignore_index=True)
repos_df.to_csv('./repos.csv')

train_samples = repos_df[repos_df['train']]['samples'].sum()
test_samples = repos_df[~repos_df['train']]['samples'].sum()
train_repos = len(repos_df[repos_df['train']]['samples'])
test_repos = len(repos_df[~repos_df['train']]['samples'])
print(f'train samples: {train_samples} | test samples: {test_samples} | ratio: {test_samples / (test_samples + train_samples):.2f} | train repos: {train_repos} | test_repos: {test_repos}')
