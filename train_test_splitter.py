import pandas as pd
from collections import Counter
from data_reader import GetDataAsPython

storage_directory = './storage/'
data = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_repo_specific_final.json")
data_eslint = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_eslint_final.json")

repos = Counter([item.repo for item in data])
sorted_repos = sorted(repos.items(), key=lambda d: d[1], reverse=True)


# In[3]:


print(sorted_repos[:10])


# In[4]:


print(len(sorted_repos), sum([value for key,value in sorted_repos]))


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


plt.hist([value for key,value in sorted_repos][:30])


# In[7]:


print(sorted_repos[:26])


# In[8]:


target_big = sorted_repos[:26:3]
print(target_big, len(target_big), sum([value for key,value in target_big]))


# In[9]:


source_big = [item for item in sorted_repos[:26] if item not in target_big]
print(source_big[:3], len(source_big), sum([value for key,value in source_big]))


# In[10]:


target_small = sorted_repos[26::7]
target_small[:2], len(target_small), sum([value for key,value in target_small])


# In[11]:


source_small = [item for item in sorted_repos[26:] if item not in target_small]
print(source_small[:3], len(source_small), sum([value for key,value in source_small]))


# In[12]:


source_small_df = pd.DataFrame(source_small, columns=['repo', 'samples'])
source_small_df['category'] = 'source'
source_small_df['size'] = 'small'
source_big_df = pd.DataFrame(source_big, columns=['repo', 'samples'])
source_big_df['category'] = 'source'
source_big_df['size'] = 'big'
target_small_df = pd.DataFrame(target_small, columns=['repo', 'samples'])
target_small_df['category'] = 'target'
target_small_df['size'] = 'small'
target_big_df = pd.DataFrame(target_big, columns=['repo', 'samples'])
target_big_df['category'] = 'target'
target_big_df['size'] = 'big'

repos_df = pd.concat([source_small_df, source_big_df, target_small_df, target_big_df], ignore_index=True)
print(repos_df.head(1))


# In[13]:


repos_df.to_csv('./repos_3.csv')


