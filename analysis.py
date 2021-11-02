import pandas as pd

result_per_repo = pd.read_csv('./storage/results_per_repo.csv',
                              names=['repo', 'accuracy', 'n_of_w', 'time', 'model-name', 'model'])

result_per_repo[result_per_repo['model'] == './storage/checkpoint-40140']['accuracy'].mean()
result_per_repo[~(result_per_repo['model'] == './storage/checkpoint-40140')]['accuracy'].mean()

old_model = result_per_repo[~(result_per_repo['model'] == './storage/checkpoint-40140')]
new_model = result_per_repo[result_per_repo['model'] == './storage/checkpoint-40140']
compare = pd.merge(old_model, new_model, on='repo')[['repo', 'accuracy_x', 'accuracy_y']]
compare['diff'] = (compare['accuracy_x'] - compare['accuracy_y']).apply(lambda x: round(x,2))
compare = compare.sort_values(by='diff', ascending=False)
compare.to_csv('./diffs.csv', index=False)
