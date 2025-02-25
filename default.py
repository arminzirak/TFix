#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os

# In[ ]:


from IPython.display import clear_output


# In[2]:


from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import T5Config
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from transformers import set_seed


# In[3]:


from datetime import datetime
import argparse
import os
import sys

sys.path.append("./hf_transformers/")


# In[4]:


import torch

from data_reader import GetDataAsPython
from prepare_data import create_data
from prepare_data import create_dataset
from prepare_data import extract_warning_types
from utils import boolean_string
from utils import get_current_time


# In[34]:


import torch

from data_reader import GetDataAsPython
from prepare_data import create_data
from prepare_data import create_dataset
from prepare_data import extract_warning_types
from utils import boolean_string
from utils import get_current_time
import csv


# In[6]:

local = False

if local:
    storage_directory = './storage/'
    load_model = f'./{storage_directory}/checkpoint-37375'
    batch_size = 16
else:
    storage_directory = '/scratch/arminz/'
    batch_size = 64
    load_model = f'/{storage_directory}/t5-small_global_repo-based_03-11-2021_15-28-40/checkpoint-37375/'

# In[7]:


import random


# In[8]:


exec_number = random.randint(0, 1000)
exec_number


# In[31]:


parser = argparse.ArgumentParser()
parser.add_argument("-tuning_results.csv", "--repo", type=str, default='/data/all/data/oroinc/platform')
parser.add_argument("-p", "--percent", type=float, default=0.2)

args = parser.parse_args()
repo = args.repo
sample_percent = args.percent

print('start:', repo, sample_percent)


# In[35]:


name='default'
name


# In[36]:


# Read and prepare data
data = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_repo_specific_final.json")
data_eslint = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_eslint_final.json")
data += data_eslint


# In[37]:


len(data)


# In[38]:


all_warning_types = extract_warning_types(data)


# In[39]:


(train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels, train_info, val_info, test_info, ) =    create_data(data, all_warning_types, include_warning=True, design='repo-based-included', select_repo=repo)


# In[40]:


tokenizer = T5Tokenizer.from_pretrained(load_model)


# In[41]:


len(train_inputs)


# In[42]:


# Create dataset required by pytorch
samples = int(sample_percent * len(train_inputs))
print(f'{len(train_inputs)} {samples} {sample_percent}')
train_dataset = create_dataset(train_inputs[:samples], train_labels[:samples], tokenizer, pad_truncate=True, max_length=128)
val_dataset = create_dataset(val_inputs[:samples], val_labels[:samples], tokenizer, pad_truncate=True)


# In[61]:


now = datetime.now()
full_name = f'{name}_{exec_number}_{repo.rsplit("/", 1)[1][-20:]}_{sample_percent}'
model_directory = f'{storage_directory}/tmp/{full_name}'
model_directory


# In[62]:

lr = 1e-4
wd = 0
ws = 500
print('default arguments', lr, wd, ws)



# In[65]:


tokenizer = T5Tokenizer.from_pretrained(load_model)
model = T5ForConditionalGeneration.from_pretrained(load_model)
model.resize_token_embeddings(len(tokenizer))
# model.eval()


# In[66]:


training_args = Seq2SeqTrainingArguments(
    output_dir=model_directory,
    num_train_epochs=30,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
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


# In[67]:


from sklearn.metrics import accuracy_score
import numpy as np
def compute_metrics(p):
    target_max_length = 256
    predictions, labels = p.predictions, p.label_ids
    
    predictions = np.pad(predictions, ((0, 0), (0, target_max_length - predictions.shape[1])), mode="constant")
    predictions = np.delete(predictions, 0, axis=1)
    predictions = np.insert(predictions, target_max_length - 1, 0, axis=1)

    

    labels = np.array(labels)
    labels = np.pad(labels, ((0, 0), (0, target_max_length - labels.shape[1])), mode="constant")
    labels = np.delete(labels, 0, axis=1)
    labels = np.insert(labels, target_max_length - 1, 0, axis=1)
    

    correct_counter = np.sum(np.all(np.equal(labels, predictions), axis=1))
    return {'acc': int(correct_counter)}


# In[68]:


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=[torch.optim.Adam(params=model.parameters(), lr=lr), None],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# In[69]:


trainer.train()


# In[73]:


trainer.evaluate()['eval_loss']


# In[77]:


best_model_dir = f'{model_directory}/best'
trainer.save_model(best_model_dir)


# In[78]:


result = os.system(f'python hf_transformers/tfix_testing.py --load-model {best_model_dir} -bs 16 --model-name t5-small -d repo-based-included -tuning_results.csv {repo}')
print(result)

# In[ ]:




