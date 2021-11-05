import pandas as pd

result_per_repo = pd.read_csv('./storage/results_per_repo.csv',
                              names=['repo', 'accuracy', 'warnings', 'samples', 'time', 'model-name', 'model'])
result_per_repo = result_per_repo[(result_per_repo['samples'] >= 20)] #27 is a good split
models = result_per_repo['model'].unique()
for model in models:
    this_model = result_per_repo[result_per_repo['model'] == model]
    accuracy = sum(this_model['accuracy']/ len(this_model))
    weighted_accuracy = sum((this_model['accuracy'] * this_model['samples']))/sum(this_model['samples'])
    print(model, f'A: {accuracy:.2f}, WA: {weighted_accuracy:.2f}')

models = ['./storage/checkpoint-38500/', './storage/checkpoint-37375/']
old_model = result_per_repo[result_per_repo['model'] == models[0]]
new_model = result_per_repo[result_per_repo['model'] == models[1]]
assert (old_model.drop(['accuracy', 'time', 'model'], axis=1).values == new_model.drop(['accuracy', 'time', 'model'], axis=1).values).all()
merged = pd.merge(old_model, new_model, on='repo')
assert (merged['samples_x'] == merged['samples_y']).all()
print('all checks passed!')
compare = merged[['repo', 'accuracy_x', 'accuracy_y', 'samples_x', 'warnings_x']]
compare['diff'] = (compare['accuracy_x'] - compare['accuracy_y']).apply(lambda x: round(x,2))
compare = compare.sort_values(by='diff', ascending=False)
compare.to_csv('./diffs.csv', index=False)
