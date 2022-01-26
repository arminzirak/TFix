#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

sys.path.append("./hf_transformers/")

# In[2]:


from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from module_utils import AdapterT5Block
import torch.nn as nn
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

from datetime import datetime
import argparse
import os

# In[3]:


start_all = datetime.now()
print(f'start all: {start_all}')

# In[ ]:


# In[4]:


parser = argparse.ArgumentParser()
parser.add_argument("-r", "--repo", type=str, default='/data/all/data/qooxdoo/qooxdoo')
parser.add_argument("-p", "--percent", type=float, default=1)
parser.add_argument("-f", type=str, required=False)

args = parser.parse_args()
repo = args.repo
sample_percent = args.percent
print(repo, sample_percent)

# In[5]:


model_name = 't5-small'

# In[6]:


local = True

if local:
    storage_directory = './storage/'
    base_model = f'./{storage_directory}/training/checkpoint-37375'
    adapted_model_dir = f'{storage_directory}/tmp/adapted'
    batch_size = 16
else:
    storage_directory = '/scratch/arminz/'
    base_model = f'./{storage_directory}/training/t5-small_repo-based_21-01-2022_10-29-42/checkpoint-16440'
    adapted_model_dir = f'./{storage_directory}/tmp/adapted'
    batch_size = 128

# In[7]:

tokenizer = T5Tokenizer.from_pretrained(base_model)
model = T5ForConditionalGeneration.from_pretrained(base_model)
model.resize_token_embeddings(len(tokenizer))
model.to('cuda')

# In[8]:


model.config.adapter_size = 50
print('adapter size:', model.config.adapter_size)

# In[9]:


import copy

encoder_config = copy.deepcopy(model.config)
encoder_config.is_decoder = False
encoder_config.use_cache = False
encoder_config.is_encoder_decoder = False
encoder_blocks = nn.ModuleList(
    [AdapterT5Block(encoder_config, has_relative_attention_bias=bool(i == 0)) for i in range(model.config.num_layers)]
)

# In[10]:


decoder_config = copy.deepcopy(model.config)
decoder_config.is_decoder = True
decoder_config.is_encoder_decoder = False
decoder_config.num_layers = model.config.num_decoder_layers
decoder_blocks = nn.ModuleList(
    [AdapterT5Block(decoder_config, has_relative_attention_bias=bool(i == 0)) for i in range(model.config.num_layers)]
)

# In[11]:


for i in range(len(encoder_blocks)):
    encoder_blocks[i].load_state_dict(model.encoder.block[i].state_dict(), strict=False)
    model.encoder.block[i] = encoder_blocks[i].to('cuda')

# In[12]:

for i in range(len(decoder_blocks)):
    decoder_blocks[i].load_state_dict(model.decoder.block[i].state_dict(), strict=False)
    model.decoder.block[i] = decoder_blocks[i].to('cuda')

# In[13]:

# tokens = tokenizer(['hi how are you', 'thanks Im fine'], padding=True, return_tensors='pt').to('cuda')
# tokens


# In[14]:


data = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_repo_specific_final.json")
data_eslint = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_eslint_final.json")
data += data_eslint
len(data)

# In[15]:


all_warning_types = extract_warning_types(data)

# In[16]:


(repo_train_inputs, repo_train_labels, repo_val_inputs, repo_val_labels, repo_test_inputs, repo_test_labels,
 repo_train_info, repo_val_info, repo_test_info,) = create_data(data, all_warning_types, include_warning=True,
                                                                design='repo-based-included', select_repo=repo)

# In[16]:


name = 'adapterTuned'

# In[17]:


exec_number = random.randint(0, 1000)
print('exec_number:', exec_number)
tokenizer = T5Tokenizer.from_pretrained(base_model)

# In[48]:


train_dataset = create_dataset(repo_train_inputs, repo_train_labels, tokenizer, pad_truncate=True, max_length=128)
val_dataset = create_dataset(repo_val_inputs, repo_val_labels, tokenizer, pad_truncate=True)
# test_dataset = create_dataset(repo_val_inputs, repo_val_labels, tokenizer, pad_truncate=True)
# 
# In[49]:


now = datetime.now()
test_result_directory = f'{storage_directory}/{name}'
full_name = f'{name}_{exec_number}_{repo.rsplit("/", 1)[1][-20:]}_{1.0}_{len(repo_train_inputs)}'
model_directory = f'{storage_directory}/tmp/{full_name}'
print('saved model directory:', model_directory)

# In[18]:


lr = 4e-3
ws = 300
wd = 0.4

# In[19]:


for param in model.parameters():
    param.requires_grad = False

for block in model.encoder.block:
    for param in block.adapter.parameters():
        param.requires_grad = True

for block in model.decoder.block:
    for param in block.adapter.parameters():
        param.requires_grad = True

# In[20]:


freezed, non_freezed = 0, 0
for item in model.parameters():
    if item.requires_grad:
        non_freezed += item.numel()
    else:
        freezed += item.numel()
print(f'percentage of learnable parameters: {non_freezed / (freezed + non_freezed):.2f}')

# In[21]:


from transformers import EarlyStoppingCallback
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import T5Config
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from transformers import set_seed

training_args = Seq2SeqTrainingArguments(
    output_dir=model_directory,
    num_train_epochs=70,
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

# In[22]:


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

# In[23]:


start_tuning = datetime.now()
print(f'start tuning: {start_all}')

# In[24]:


trainer.train()

# In[25]:


end_tuning = datetime.now()
print(f'end tuining: {start_all}')

# In[26]:


print('eval', trainer.evaluate()['eval_loss'])

# In[27]:


model.save_pretrained(adapted_model_dir)
tokenizer.save_pretrained(adapted_model_dir)

# In[29]:


for i, block in enumerate(model.encoder.block):
    torch.save(block.adapter.state_dict(), f'{adapted_model_dir}/adapter-encoder-{i}')

# In[30]:


for i, block in enumerate(model.decoder.block):
    torch.save(block.adapter.state_dict(), f'{adapted_model_dir}/adapter-decoder-{i}')

# In[31]:


end_all = datetime.now()
print(f'end all: {start_all}')

# In[32]:


import csv

with open('tuner_runtime.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(
        [name, repo, len(train_dataset), len(val_dataset), base_model, start_all, start_tuning, end_tuning, end_all])

# In[33]:


from numba import cuda

device = cuda.get_current_device()
device.reset()

# In[34]:


result = os.system(
    f'python hf_transformers/tfix_testing_adapterLayer.py --load-model {adapted_model_dir} -bs {batch_size} --model-name {model_name} -d repo-based-included -r {repo}')
print(result)

# In[36]:


import shutil

shutil.rmtree(adapted_model_dir)
#


# In[ ]:
