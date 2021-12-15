#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append("./hf_transformers/")


# In[3]:


from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import T5Config
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from transformers import set_seed


# In[4]:


from datetime import datetime
import argparse
import os


# In[5]:


import torch

from data_reader import GetDataAsPython
from prepare_data import create_data
from prepare_data import create_dataset
from prepare_data import extract_warning_types
from utils import boolean_string
from utils import get_current_time
import csv
import numpy as np
import random


# In[6]:


local = True

if local:
    storage_directory = './storage/'
    base_model = f'./{storage_directory}/checkpoint-37375'
    batch_size = 16
    codebert_address = "microsoft/codebert-base"
else:
    storage_directory = '/scratch/arminz/'
    batch_size = 64
    # base_model = f'/{storage_directory}/t5-small_global_repo-based_03-11-2021_15-28-40/checkpoint-37375/'
    base_model = f'{storage_directory}/checkpoint-37375'
    codebert_address = "/home/arminz/codebert-base"


# In[7]:


import codebert_utils
codebert_utils.load(codebert_address)


# In[8]:


exec_number = random.randint(0, 1000)

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--append", type=int, required=True)
parser.add_argument("-rp", "--repo_percent", type=float, required=True)
parser.add_argument("-r", "--repo", type=str, required=True)

args = parser.parse_args()
append = args.append
repo = args.repo
repo_percent = args.repo_percent


# In[9]:


data = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_repo_specific_final.json")
data_eslint = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_eslint_final.json")
data += data_eslint


# In[10]:


name = 'added'


# In[11]:


all_warning_types = extract_warning_types(data)

# In[16]:


(repo_train_inputs, repo_train_labels, repo_val_inputs, repo_val_labels, repo_test_inputs, repo_test_labels,
 repo_train_info, repo_val_info, repo_test_info,) = create_data(data, all_warning_types, include_warning=True,
                                                                design='repo-based-included', select_repo=repo)

# In[17]:


(general_train_inputs, general_train_labels, general_val_inputs, general_val_labels, general_test_inputs,
 general_test_labels, general_train_info, general_val_info, general_test_info,) = create_data(data, all_warning_types,
                                                                                              include_warning=True,
                                                                                              design='repo-based')


# In[12]:


repo_vecs = np.array([codebert_utils.code_to_vec(item) for item in repo_train_inputs])


# In[13]:


from sklearn.neighbors import NearestNeighbors


# In[14]:


general_vecs = np.load('general_arr_all.npy')

# In[28]:


nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(repo_vecs)


# In[15]:


distances, indices = nbrs.kneighbors(general_vecs)


# In[16]:


repo_nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(repo_vecs)
repo_distances, index = repo_nbrs.kneighbors(repo_vecs)


# In[17]:


threshold = sorted(distances)[int(append * 5 / 4)]
print('threshold:', threshold, 'append:', append, int(append * 5 / 4))


# In[18]:


selected = (distances < threshold)


# In[19]:


assert selected.sum() == int(append * 5 / 4)


# In[20]:


samples = int(repo_percent * len(repo_train_inputs))
print(f'all repo samples: {len(repo_train_inputs)}\ntrain samples of repo: {samples} ({repo_percent})')


# In[21]:


assert len(distances) == len(general_train_inputs)


# In[22]:


filtered_general_inputs = list()
filtered_general_labels = list()
filtered_general_info = list()
for ind in range(len(general_train_inputs)):
    if selected[ind]:
        filtered_general_inputs.append(general_train_inputs[ind])
        filtered_general_labels.append(general_train_labels[ind])
        filtered_general_info.append(general_train_info[ind])


# In[23]:


selected.shape


# In[24]:


filtered_general_distances, _ = nbrs.kneighbors(general_vecs[selected[:, 0]])
filtered_general_distances.shape


# In[25]:


# from matplotlib import pyplot as plt
# plt.boxplot([distances.squeeze(), repo_distances[:, 1], filtered_general_distances.squeeze()])


# In[26]:


print(len(filtered_general_inputs))
assert len(filtered_general_inputs) == int(append * 5 / 4)


# In[27]:


validation_point = append
print(validation_point)


# In[28]:


added_inputs_train = filtered_general_inputs[:validation_point]
added_labels_train = filtered_general_labels[:validation_point]
added_info_train = filtered_general_info[:validation_point]


# In[29]:


added_inputs_val = filtered_general_inputs[validation_point:]
added_labels_val = filtered_general_labels[validation_point:]
added_info_val = filtered_general_info[validation_point:]


# In[30]:


print(len(repo_train_inputs), len(added_inputs_train))


# In[31]:


assert len(added_inputs_train) == append


# In[32]:


added_inputs_train += repo_train_inputs[:samples]
added_labels_train += repo_train_labels[:samples]
added_info_train += repo_train_info[:samples]


# In[33]:


added_inputs_val += repo_val_inputs
added_labels_val += repo_val_labels
added_info_val += repo_val_info


# In[34]:


print('added inputs train', len(added_inputs_train))
print('added inputs val', len(added_inputs_val))


# In[35]:


print('repo val inputs', len(repo_val_inputs))


# In[36]:


assert len(added_inputs_train) == append + samples


# In[37]:


tokenizer = T5Tokenizer.from_pretrained(base_model)

# In[48]:


train_dataset = create_dataset(added_inputs_train, added_labels_train, tokenizer, pad_truncate=True, max_length=128)
val_dataset = create_dataset(added_inputs_val, added_labels_val, tokenizer, pad_truncate=True)
test_dataset = create_dataset(repo_val_inputs, repo_val_labels, tokenizer, pad_truncate=True)

# In[49]:


now = datetime.now()
test_result_directory = f'{storage_directory}/fine-tune-result'
full_name = f'{name}_{exec_number}_{repo.rsplit("/", 1)[1][-20:]}_{repo_percent}_{samples}_{append}'
model_directory = f'{storage_directory}/tmp/{full_name}'
model_directory


# In[38]:


len(repo_test_inputs)


# In[39]:


lr = 4e-3
ws = 300
wd = 0.4


# In[40]:


tokenizer = T5Tokenizer.from_pretrained(base_model)
model = T5ForConditionalGeneration.from_pretrained(base_model)
model.resize_token_embeddings(len(tokenizer))


# In[41]:



from transformers import EarlyStoppingCallback

training_args = Seq2SeqTrainingArguments(
    output_dir=model_directory,
    num_train_epochs=15,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=ws,
    weight_decay=wd,
    logging_dir=model_directory,
    logging_steps=100,
    do_eval=True,
    evaluation_strategy="epoch",
    learning_rate=lr,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=1,
    eval_accumulation_steps=1,  # set this lower, if testing or validation crashes
    disable_tqdm=False,
    predict_with_generate=True,  # never set this to false.
    seed=42,  # default value
)


# In[42]:


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=[torch.optim.Adam(params=model.parameters(), lr=lr), None],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
    #     compute_metrics=compute_metrics
)


# In[ ]:


trainer.train()


# In[ ]:


print('eval', trainer.evaluate()['eval_loss'])


# In[ ]:


best_model_dir = f'{model_directory}/best/'
trainer.save_model(best_model_dir)
print('best model dir', best_model_dir)


# In[42]:


from numba import cuda 
device = cuda.get_current_device()
device.reset()


# In[43]:



os.system(
    f'python hf_transformers/tfix_testing.py --load-model {best_model_dir} -bs {batch_size} --model-name t5-small -d repo-based-included -r {repo}')


# In[44]:


import shutil


# In[45]:


shutil.rmtree(model_directory)


# In[ ]:




