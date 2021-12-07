#!/usr/bin/env python
# coding: utf-8

# In[2]:

#
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[3]:


# In[4]:


from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import T5Config
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from transformers import set_seed

# In[5]:


from datetime import datetime
import argparse
import os
import sys

sys.path.append("./hf_transformers/")

# In[6]:


import torch

from data_reader import GetDataAsPython
from prepare_data import create_data
from prepare_data import create_dataset
from prepare_data import extract_warning_types
from utils import boolean_string
from utils import get_current_time
import csv
import numpy as np

# In[7]:

local = False

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

# In[8]:


import random

# In[9]:


exec_number = random.randint(0, 1000)
exec_number

# In[10]:

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--append", type=int, required=True)
parser.add_argument("-rp", "--repo_percent", type=float, required=True)
parser.add_argument("-r", "--repo", type=str, required=True)

args = parser.parse_args()
append = args.append
repo = args.repo
repo_percent = args.repo_percent

# repo = '/data/all/data/appium/appium'
# repo


# In[11]:


# repo_percent = 0.0
print('repo percent:', repo_percent)

# In[12]:


name = 'added'
name

# In[13]:


# Read and prepare data
data = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_repo_specific_final.json")
data_eslint = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_eslint_final.json")
data += data_eslint

# In[14]:


len(data)

# In[15]:


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

# In[18]:

from transformers import AutoTokenizer, AutoModel

code_bert_model = AutoModel.from_pretrained(codebert_address)
code_bert_model.to('cuda')

code_bert_tokenizer = AutoTokenizer.from_pretrained(codebert_address)


# In[19]:


def code_to_vec(code):  # probably need a normalization
    code_tokens = code_bert_tokenizer.tokenize(code)
    tokens = [code_bert_tokenizer.cls_token] + code_tokens + [code_bert_tokenizer.sep_token]
    tokens_ids = code_bert_tokenizer.convert_tokens_to_ids(tokens)
    context_embeddings = code_bert_model(torch.tensor(tokens_ids).to('cuda')[None, :])[0]
    return context_embeddings[0][0].cpu().detach().numpy()


def vec_distance(code1, code2):
    n_code1 = code1 / np.linalg.norm(code1)
    n_code2 = code1 / np.linalg.norm(code2)
    return np.linalg.norm(n_code1 - n_code2)


# In[20]:


repo_vecs = np.array([code_to_vec(item) for item in repo_train_inputs])

# In[21]:


repo_vecs.shape

# In[22]:


from sklearn.neighbors import NearestNeighbors

# In[23]:


# repo_center = np.average(repo_vecs, axis=0)


# In[24]:


# %%time
# general_vecs = np.array([code_to_vec(item[:512]) for item in general_train_inputs[:]])


# In[25]:


# general_vecs.shape


# In[26]:


# np.save('general_arr_all.npy', general_vecs)


# In[27]:


general_vecs = np.load('general_arr_all.npy')

# In[28]:


nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(repo_vecs)

# In[29]:


distances, indices = nbrs.kneighbors(general_vecs)

# In[30]:


# diffs = [vec_distance(general_vec, repo_center) for general_vec in general_vecs]


# In[31]:


# In[32]:


repo_nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(repo_vecs)
repo_distances, index = repo_nbrs.kneighbors(repo_vecs)
repo_distances.shape

# In[33]:


# In[34]:


# top_n = 597#int(len(diffs) / 10)
# top_n


# In[38]:
threshold = sorted(distances)[append]
print('threshold:', threshold, 'append:', append)

selected = (distances < threshold)
# selected.sum()


# In[39]:


# selected_ind = np.argpartition(diffs, top_n)[:top_n]


# In[40]:


# np.array(diffs)[selected_ind].mean(), np.array(diffs).mean()


# In[41]:


added_inputs = list()
added_labels = list()
added_info = list()
for ind in range(len(general_train_inputs)):
    if selected[ind]:
        added_inputs.append(general_train_inputs[ind])
        added_labels.append(general_train_labels[ind])
        added_info.append(general_train_info[ind])

# In[42]:


# Create dataset required by pytorch
samples = int(repo_percent * len(repo_train_inputs))
print(f'{len(repo_train_inputs)} {samples} {repo_percent}')

# In[43]:


print(len(repo_train_inputs), len(added_inputs))

# In[44]:


added_inputs += repo_train_inputs[:samples]
added_labels += repo_train_labels[:samples]
added_info += repo_train_info[:samples]

# In[45]:

print('added inputs finally', len(added_inputs))
validation_point = int((len(added_inputs) * 4) / 5)

# In[46]:


len(added_inputs)

# In[47]:


tokenizer = T5Tokenizer.from_pretrained(base_model)

# In[48]:


train_dataset = create_dataset(added_inputs[:validation_point], added_labels[:validation_point], tokenizer,
                               pad_truncate=True, max_length=128)
val_dataset = create_dataset(added_inputs[validation_point:], added_labels[validation_point:], tokenizer,
                             pad_truncate=True)
test_dataset = create_dataset(repo_val_inputs, repo_val_labels, tokenizer, pad_truncate=True)

# In[49]:


now = datetime.now()
test_result_directory = f'{storage_directory}/fine-tune-result'
full_name = f'{name}_{exec_number}_{repo.rsplit("/", 1)[1][-20:]}_{repo_percent}_{selected.sum()}'
model_directory = f'{storage_directory}/tmp/{full_name}'
model_directory

# In[50]:


lr = 4e-3
ws = 300
wd = 0.4
lr, wd, ws

# In[51]:


tokenizer = T5Tokenizer.from_pretrained(base_model)
model = T5ForConditionalGeneration.from_pretrained(base_model)
model.resize_token_embeddings(len(tokenizer))
# model.eval()


# In[52]:


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

# In[53]:


# from sklearn.metrics import accuracy_score
# import numpy as np
# def compute_metrics(p):
#     target_max_length = 256
#     predictions, labels = p.predictions, p.label_ids

#     predictions = np.pad(predictions, ((0, 0), (0, target_max_length - predictions.shape[1])), mode="constant")
#     predictions = np.delete(predictions, 0, axis=1)
#     predictions = np.insert(predictions, target_max_length - 1, 0, axis=1)

#     labels = np.array(labels)
#     labels = np.pad(labels, ((0, 0), (0, target_max_length - labels.shape[1])), mode="constant")
#     labels = np.delete(labels, 0, axis=1)
#     labels = np.insert(labels, target_max_length - 1, 0, axis=1)

#     correct_counter = np.sum(np.all(np.equal(labels, predictions), axis=1))
#     return {'acc': int(correct_counter)}


# In[54]:


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=[torch.optim.Adam(params=model.parameters(), lr=lr), None],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    #     compute_metrics=compute_metrics
)

# In[55]:


trainer.train()

# In[56]:


print('eval', trainer.evaluate()['eval_loss'])

# In[57]:


trainer.predict(test_dataset).metrics

# In[58]:


best_model_dir = f'{model_directory}/best/'
trainer.save_model(best_model_dir)
print('best model dir', best_model_dir)

# In[59]:


# del train_dataset
# del test_dataset
# del val_dataset
# del trainer


# In[60]:


# In[61]:


os.system(
    f'python hf_transformers/tfix_testing.py --load-model {best_model_dir} -bs {batch_size} --model-name t5-small -d repo-based-included -r {repo}')

import shutil

shutil.rmtree(model_directory)

