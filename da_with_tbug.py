#!/usr/bin/env python
# coding: utf-8

# In[ ]:



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

from data_reader import GetDataAsPython, MinimalDataPoint
from prepare_data import create_data
from prepare_data import create_dataset
from prepare_data import extract_warning_types
from utils import boolean_string
from utils import get_current_time
import csv

start_all = datetime.now()

# In[6]:

import socket
local = False if 'computecanada' in socket.gethostname() else True

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--repo", type=str, default='/data/all/data/oroinc/platform')
parser.add_argument("-l", "--large", action="store_true")
# parser.add_argument("-p", "--percent", type=float, default=1)

args = parser.parse_args()
repo = args.repo
large = args.large
#sample_percent = args.percent

print('start:', repo)

lr = 4e-3
ws = 300
wd = 0.4
print('best arguments', lr, wd, ws)


if not large:
    base_model = 'training/t5-small_repo-based_21-01-2022_10-29-42/checkpoint-16440'
else:
    base_model = 'training/t5-large_global_repo-based_25-01-2022_10-31-49/checkpoint-218825'

if local:
    storage_directory = './storage/'
    load_model = f'./{storage_directory}/{base_model}'
    batch_size = 16
else:
    storage_directory = '/scratch/arminz/'
    batch_size = 64
    load_model = f'/{storage_directory}/{base_model}'

# In[7]:


# In[3]:




# In[4]:



import random


# In[8]:


exec_number = random.randint(0, 1000)
exec_number


# In[31]:







# In[35]:


name='good'
name


# In[5]:


model_name = 't5-small' if not large else 't5-large'


# In[6]:


# Read and prepare data
test_data = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_repo_specific_final.json")
test_data_eslint = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_eslint_final.json")
test_data += test_data_eslint


# In[7]:


import json
with open(f'{storage_directory}/bt_data_{model_name}/scores.json', 'r') as f:
    scores = json.load(f)
scores


# In[20]:
all_warning_types = extract_warning_types(test_data)


good_warnings = [key for key in scores if scores[key] != 'NA' and float(scores[key]) >= 0.1]
# good_warnings = all_warning_types
print(good_warnings)



# In[ ]:





# In[21]:


len(test_data)


# In[38]:




# In[39]:


(_train_inputs, _train_labels, _val_inputs, _val_labels, test_inputs, test_labels, train_info, val_info, test_info, ) =     create_data(test_data, good_warnings, include_warning=True, design='repo-based-included', select_repo=repo)


# In[11]:


train_data = MinimalDataPoint.FromJsonToPython(f'{storage_directory}/bt_data_{model_name}/{repo}.json')


# In[23]:
train_data = [item for item in train_data if item.correct]

(train_inputs, train_labels, val_inputs, val_labels, _test_inputs, _test_labels, train_info, val_info, test_info, ) =     create_data(train_data, good_warnings, include_warning=True, design='repo-based-included', select_repo=repo)


# In[24]:


# train_labels[0]


# In[25]:



tokenizer = T5Tokenizer.from_pretrained(load_model)


# In[41]:



# In[26]:



# Create dataset required by pytorch
# samples = int(sample_percent * len(train_inputs))
train_dataset = create_dataset(train_inputs[:], train_labels[:], tokenizer, pad_truncate=True, max_length=128)
val_dataset = create_dataset(val_inputs, val_labels, tokenizer, pad_truncate=True)

# print(f'amount of data that is being used for fine-tuning (train) : {len(train_dataset)} == {samples} ({sample_percent})')
print(f'amount of data that is being used for fine-tuning (validation): {len(val_dataset)} (full)')
print(f'amount of data that will be probably being used for testing: {sum([len(x) for x in test_inputs.values()])} (full)')


# In[27]:


now = datetime.now()
full_name = f'{name}_{exec_number}_{repo.rsplit("/", 1)[1][-20:]}'
model_directory = f'{storage_directory}/tmp/bt_{model_name}/{full_name}'
model_directory


# In[28]:


tokenizer = T5Tokenizer.from_pretrained(load_model)
model = T5ForConditionalGeneration.from_pretrained(load_model)
model.resize_token_embeddings(len(tokenizer))
model.to('cuda')


# In[24]:



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


# In[25]:



# In[68]:
from transformers import EarlyStoppingCallback


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=[torch.optim.Adam(params=model.parameters(), lr=lr), None],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
)


# In[26]:


trainer.train()


# In[27]:


trainer.evaluate()


# In[28]:


model_directory


# In[29]:


tuned_model_dir=f'{storage_directory}/tmp/bt_{model_name}/' + repo
trainer.save_model(tuned_model_dir)


# In[30]:


# os.system(f'python hf_transformers/tfix_testing.py --load-model {tuned_model_dir} -bs 8 --model-name {model_name} -d repo-based-included -r {repo}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




