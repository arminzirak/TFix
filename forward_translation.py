#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


SCORE_THRESHHOLD = 0
FW_EPOCHS = 1

repo_list = ['/data/all/data/qooxdoo/qooxdoo', '/data/all/data/elastic/kibana', '/data/all/data/zloirock/core-js', '/data/all/data/Encapsule-Annex/onm', '/data/all/data/emberjs/ember.js', '/data/all/data/sequelize/sequelize',
             '/data/all/data/dcos/dcos-ui', '/data/all/data/LivelyKernel/LivelyKernel', '/data/all/data/svgdotjs/svg.js', '/data/all/data/foam-framework/foam']

# repo = '/data/all/data/emberjs/ember.js'


# In[3]:


import sys
sys.path.append("./hf_transformers/")


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
import random


# In[7]:


import socket
local = False if 'computecanada' in socket.gethostname() else True

if local:
    storage_directory = './storage/'
    base_model = f'./{storage_directory}/training//t5-small_repo-based_21-01-2022_10-29-42/checkpoint-16440'
    batch_size = 16
#     codebert_address = "microsoft/codebert-base"
else:
    storage_directory = '/scratch/arminz/'
    batch_size = 64
    # base_model = f'/{storage_directory}/t5-small_global_repo-based_03-11-2021_15-28-40/checkpoint-37375/'
    base_model = f'{storage_directory}/training/checkpoint-37375'
#     codebert_address = "/home/arminz/codebert-base"


# In[8]:


# import codebert_utils
# codebert_utils.load(codebert_address)


# In[9]:


exec_number = random.randint(0, 1000)

parser = argparse.ArgumentParser()
# parser.add_argument("-a", "--append", type=int, required=True)
# parser.add_argument("-rp", "--repo_percent", type=float, required=True)
parser.add_argument("-r", "--repo", type=str, required=True)

args = parser.parse_args()
# append = 56*1000#args.append
repo = args.repo
# repo_percent = 0 # args.repo_percent


# In[10]:


data = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_repo_specific_final.json")
data_eslint = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_eslint_final.json")
data += data_eslint
len(data)


# In[11]:


data[0].repo


# In[12]:


len(data)


# In[14]:


name = 'ft1'


# In[15]:


all_warning_types = extract_warning_types(data)

# In[16]:

(repo_train_inputs, repo_train_labels, repo_val_inputs, repo_val_labels, repo_test_inputs, repo_test_labels,
 repo_train_info, repo_val_info, repo_test_info,) = create_data(data, all_warning_types, include_warning=True,
                                                                design='repo-based-included', select_repo=repo)

#


# In[16]:


tokenizer = T5Tokenizer.from_pretrained(base_model)

# In[48]:


train_dataset = create_dataset(repo_train_inputs, repo_train_labels, tokenizer, pad_truncate=True, max_length=128)
val_dataset = create_dataset(repo_val_inputs, repo_val_labels, tokenizer, pad_truncate=True)
# test_dataset = create_dataset(repo_val_inputs, repo_val_labels, tokenizer, pad_truncate=True)

# In[49]:


now = datetime.now()
test_result_directory = f'{storage_directory}/fine-tune-result'
full_name = f'{name}_{exec_number}_{repo.rsplit("/", 1)[1][-20:]}_{SCORE_THRESHHOLD}_{FW_EPOCHS}'
model_directory = f'{storage_directory}/tmp/{full_name}'
model_directory


# In[17]:


len(repo_test_inputs)


# In[18]:


lr = 1e-3#4e-3
ws = 300
wd = 0.4


# In[19]:


tokenizer = T5Tokenizer.from_pretrained(base_model)
model = T5ForConditionalGeneration.from_pretrained(base_model)
model.resize_token_embeddings(len(tokenizer))
model.to('cuda')


# In[20]:


from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)#, collate_fn=data_collator)
eval_dataloader = DataLoader(val_dataset, batch_size)#, collate_fn=data_collator)


# In[21]:


from torch.nn.functional import softmax
from transformers import EarlyStoppingCallback


for fw_epoch in range(FW_EPOCHS):
    if fw_epoch > 0:
        model = T5ForConditionalGeneration.from_pretrained(f'{storage_directory}/tmp_test_model').to('cuda')

    print('---- ' + str(fw_epoch))
    predictions_all = []
    scores = []
    for batch in train_dataloader:
        batch = {k: v.to('cuda') for k, v in batch.items()}
        outputs = model(**batch)
    #     loss = outputs[0]#outputs.loss
    #     loss.backward()
    #     optimizer.step()
    #     lr_scheduler.step()
    #     optimizer.zero_grad()
    #     progress_bar.update(1)

        predictions = outputs[1].argmax(-1)
        for prediction in predictions:
            decoded = tokenizer.decode(prediction)
            predictions_all.append(decoded[:decoded.find('<pad>')])
        scores += [item.item() for item in softmax(outputs[1], dim=-1).max(-1).values.prod(-1).to('cpu')]

    print(min(scores), max(scores), sum(scores) / len(scores))

    filterred_repo_train_inputs, filterred_predictions, filtered_scores = list(), list(), list()
    for repo_train_input, prediction, score in zip(repo_train_inputs, predictions_all, scores):
        if score > SCORE_THRESHHOLD:
            filterred_repo_train_inputs.append(repo_train_input)
            filterred_predictions.append(prediction)
            filtered_scores.append(score)
    print(len(filtered_scores), sum(filtered_scores) / len(filtered_scores))

    created_tune_dataset = create_dataset(filterred_repo_train_inputs, filterred_predictions, tokenizer, pad_truncate=True, max_length=128)


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

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=created_tune_dataset,
        eval_dataset=val_dataset,
        optimizers=[torch.optim.Adam(params=model.parameters(), lr=lr), None],
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        #     compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(f'{storage_directory}/tmp_test_model')


# In[21]:



# predictions_all = []
# scores = []
# for batch in train_dataloader:
#     batch = {k: v.to('cuda') for k, v in batch.items()}
#     outputs = model(**batch)
# #     loss = outputs[0]#outputs.loss
# #     loss.backward()
# #     optimizer.step()
# #     lr_scheduler.step()
# #     optimizer.zero_grad()
# #     progress_bar.update(1)

#     predictions = outputs[1].argmax(-1)
#     for prediction in predictions:
#         decoded = tokenizer.decode(prediction)
#         predictions_all.append(decoded[:decoded.find('<pad>')])
#         scores += [item.item() for item in softmax(outputs[1], dim=-1).max(-1).values.prod(-1).to('cpu')]


# In[22]:


# import matplotlib.pyplot as plt
# plt.boxplot(scores)


# In[23]:


# tokenizer.decode(predictions)
# predictions.shape
ind = 15
predictions_all[ind], repo_train_inputs[ind], repo_train_labels[ind]


# In[24]:


# filttered_repo_train_inputs, filterred_predictions, filttered_scores = list(), list(), list()
# for repo_train_input, prediction, score in zip(repo_train_inputs, predictions_all, scores):
#     if score > 0.4:
#         filttered_repo_train_inputs.append(repo_train_input)
#         filterred_predictions.append(prediction)
#         filttered_scores.append(score)


# In[25]:


# created_tune_dataset = create_dataset(filttered_repo_train_inputs, filterred_predictions, tokenizer, pad_truncate=True, max_length=128)


# In[26]:


# from transformers import EarlyStoppingCallback

# training_args = Seq2SeqTrainingArguments(
#     output_dir=model_directory,
#     num_train_epochs=15,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     warmup_steps=ws,
#     weight_decay=wd,
#     logging_dir=model_directory,
#     logging_steps=100,
#     do_eval=True,
#     evaluation_strategy="epoch",
#     learning_rate=lr,
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss",
#     greater_is_better=False,
#     save_total_limit=1,
#     eval_accumulation_steps=1,  # set this lower, if testing or validation crashes
#     disable_tqdm=False,
#     predict_with_generate=True,  # never set this to false.
#     seed=42,  # default value
# )


# In[27]:


# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=created_tune_dataset,
#     eval_dataset=val_dataset,
#     optimizers=[torch.optim.Adam(params=model.parameters(), lr=lr), None],
#     tokenizer=tokenizer,
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
#     #     compute_metrics=compute_metrics
# )


# In[28]:


# trainer.train()


# In[29]:


print('eval', trainer.evaluate()['eval_loss'])


# In[30]:


best_model_dir = f'{model_directory}/best/'
trainer.save_model(best_model_dir)
print('best model dir', best_model_dir)


# In[31]:


# from numba import cuda
# device = cuda.get_current_device()
# device.reset()
#
#
# # In[32]:


os.system(
    f'python hf_transformers/tfix_testing.py --load-model {best_model_dir} -bs 8 --model-name t5-small -d repo-based-included -r {repo}')


# In[33]:


import shutil


# In[34]:


shutil.rmtree(best_model_dir)


# In[35]:


shutil.rmtree(f'{storage_directory}/tmp_test_model')


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




