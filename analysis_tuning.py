import pandas as pd

result_per_repo = pd.read_csv('./tuning_results.csv',
                              names=['repo', 'accuracy', 'warnings', 'samples', 'time', 'model-name', 'model'])

types = ('good', 'best', 'default')
values = (0.1, 0.2, 0.3)
result_per_repo = result_per_repo[(result_per_repo['samples'] >= 20)] #27 is a good split
repos = result_per_repo['repo'].unique()


for value in values:
    print(value)
    defaults = list()
    goods = list()
    bests = list()
    for repo in repos:

        default = result_per_repo[(result_per_repo['repo'] == repo) &
                                  (result_per_repo['model'].apply(lambda x: str(value) in x)) &
                                  (result_per_repo['model'].apply(lambda x: 'default' in x))]['accuracy']

        best = result_per_repo[(result_per_repo['repo'] == repo) &
                                  (result_per_repo['model'].apply(lambda x: str(value) in x)) &
                                  (result_per_repo['model'].apply(lambda x: 'best_' in x))]['accuracy']

        good = result_per_repo[(result_per_repo['repo'] == repo) &
                                  (result_per_repo['model'].apply(lambda x: str(value) in x)) &
                                  (result_per_repo['model'].apply(lambda x: 'good' in x))]['accuracy']

        default = default.iloc[0] if len(default) else None
        best = best.iloc[0] if len(best) else None
        good = good.iloc[0] if len(good) else None

        if default and best and good:
            defaults.append(default)
            bests.append(best)
            goods.append(good)
            print(repo, value, default, good, best)

    cnt = len(defaults)
    print(sum(defaults) / cnt)
    print(sum(goods) / cnt)
    print(sum(bests) / cnt)
