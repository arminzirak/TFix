#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os

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

from data_reader import GetDataAsPython
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

base_model = 'training/t5-small_repo-based_21-01-2022_10-29-42/checkpoint-16440'

if local:
    storage_directory = './storage/'
    load_model = f'./{storage_directory}/{base_model}'
    batch_size = 16
else:
    storage_directory = '/scratch/arminz/'
    batch_size = 64
    load_model = f'/{storage_directory}/{base_model}'

# In[7]:


import random


# In[8]:


exec_number = random.randint(0, 1000)
exec_number


# In[31]:


parser = argparse.ArgumentParser()
parser.add_argument("-r", "--repo", type=str, default='/data/all/data/oroinc/platform')
parser.add_argument("-p", "--percent", type=float, default=1)

args = parser.parse_args()
repo = args.repo
sample_percent = args.percent

print('start:', repo, sample_percent)

lr = 4e-3
ws = 300
wd = 0.4
print('best arguments', lr, wd, ws)




# In[35]:


name='good'
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


(train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels, train_info, val_info, test_info, ) = \
    create_data(data, all_warning_types, include_warning=True, design='repo-based-included', select_repo=repo)


# In[40]:


tokenizer = T5Tokenizer.from_pretrained(load_model)


# In[41]:


len(train_inputs)


# In[42]:


# Create dataset required by pytorch
samples = int(sample_percent * len(train_inputs))
train_dataset = create_dataset(train_inputs[:samples], train_labels[:samples], tokenizer, pad_truncate=True, max_length=128)
val_dataset = create_dataset(val_inputs, val_labels, tokenizer, pad_truncate=True)

print(f'amount of data that is being used for fine-tuning (train) : {len(train_dataset)} == {samples} ({sample_percent})')
print(f'amount of data that is being used for fine-tuning (validation): {len(val_dataset)} (full)')
print(f'amount of data that will be probably being used for testing: {sum([len(x) for x in test_inputs.values()])} (full)')

# In[61]:


now = datetime.now()
full_name = f'{name}_{exec_number}_{repo.rsplit("/", 1)[1][-20:]}_{sample_percent}'
model_directory = f'{storage_directory}/tmp/finetuned/{full_name}'
model_directory


# In[62]:



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
from transformers import EarlyStoppingCallback


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=[torch.optim.Adam(params=model.parameters(), lr=lr), None],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
)


# In[69]:

start_training = datetime.now()

trainer.train()

end_training = datetime.now()


# In[73]:


print(f'final eval loss : {trainer.evaluate()["eval_loss"]}')


# In[77]:


# tuned_model_dir = f'{model_directory}/best'
tuned_model_dir=f'{storage_directory}/tmp/finetuned/' + repo
trainer.save_model(tuned_model_dir)

end_all = datetime.now()
import csv
with open('tuner_runtime.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([name, repo, len(train_dataset), len(val_dataset), base_model, start_all, start_training, end_training, end_all])

# In[78]:

if local:
    from numba import cuda
    device = cuda.get_current_device()
    device.reset()
#
#
# result = os.system(f'python hf_transformers/tfix_testing.py --load-model {tuned_model_dir} -bs 16 --model-name t5-small -d repo-based-included -r {repo}')
# print(result)
#
# result = os.system(f'python hf_transformers/tfix_testing.py --load-model {tuned_model_dir} -bs 16 --model-name t5-small -d source-test')
# print(result)
#
# #
# import shutil
#
# shutil.rmtree(tuned_model_dir)
#


