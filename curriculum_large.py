#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
#
# get_ipython().run_line_magic('autoreload', '')
import os


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
import numpy as np

sys.path.append("./hf_transformers/")


# In[3]:


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


# In[4]:
import socket
local = False if 'computecanada' in socket.gethostname() else True

base_model = 'training/t5-small_repo-based_21-01-2022_10-29-42/checkpoint-16440'

if local:
    raise Exception('You cannot run large on local')
    # storage_directory = './storage/'
    # load_model = f'./{storage_directory}/{base_model}'
    # batch_size = 16
else:
    storage_directory = '/scratch/arminz/'
    batch_size = 16
    load_model = f'/{storage_directory}/{base_model}'

# In[7]:


import random


# In[8]:


exec_number = random.randint(0, 1000)
exec_number


# In[5]:


import codebert_utils
codebert_address = "microsoft/codebert-base"
codebert_utils.load(codebert_address)


# In[6]:


general_vecs = np.load('general_arr_all_source.npy' if False else 'general_arr_all.npy')


# In[7]:


from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(general_vecs)


# In[ ]:





# In[ ]:



parser = argparse.ArgumentParser()
parser.add_argument("-r", "--repo", type=str, default='/data/all/data/oroinc/platform')
parser.add_argument("-m", "--mode", type=str, required=True, choices=['conf', 'length_label', 'length_input', 'distance_based'])
parser.add_argument("-md", "--model-address", type=str, required=True)

args = parser.parse_args()
model_address = args.model_address

repo = args.repo

sample_percent = 1#args.percent

print('start:', repo, sample_percent)

lr = 4e-3
ws = 300
wd = 0.4
print('best arguments', lr, wd, ws)




# In[35]:


name='curr'
name


# In[36]:


# Read and prepare data
data = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_repo_specific_final.json")
data_eslint = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_eslint_final.json")
data += data_eslint


len(data)


# In[ ]:



all_warning_types = extract_warning_types(data)


# In[39]:


(train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels, train_info, val_info, test_info, ) =     create_data(data, all_warning_types, include_warning=True, design='repo-based-included', select_repo=repo)


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


# In[ ]:



# len(distance_priorities), len(repo_vecs)


# In[ ]:


# from transformers import AutoTokenizer, DataCollatorWithPadding

# checkpoint = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# def tokenize_function(example):
#     return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


# tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# In[ ]:


from torch.utils.data import Sampler, SequentialSampler
from typing import Sized, Iterator
class MySampler(Sampler[int]):
    data_source: Sized
        
    def __init__(self, data_source: Sized, shuffle=False) -> None:
        self.data_source = data_source
        self.mode = 'all'
        self.priority =  np.zeros(len(data_source))
#         self.threshold = 0
        self.curriculum = 0
        self.shuffle = shuffle
    def __iter__(self) -> Iterator[int]:
        if self.mode == 'active':
#             return reversed(np.argsort(self.priority)[(np.sort(self.priority) < self.threshold).sum():])
            res = list(np.argsort(self.priority)[::-1][:int(self.curriculum * len(self.data_source))])
            if self.shuffle:
                random.shuffle(res)
            return iter(res)
        else:
            return iter(range(len(self.data_source)))
        
    def __len__(self) -> int:
#         return len(self.data_source) if self.mode == 'all' else (np.array(self.priority) > self.threshold).sum()
        return len(self.data_source) if self.mode == 'all' else int(self.curriculum * len(self.data_source))

    def set_priority(self, priority):
        self.priority = priority
        
#     def set_threshhold(self, threshold):
#         self.threshold = threshold
    def set_curriculum(self, curriculum):
        self.curriculum = curriculum
        
    def set_mode(self, mode):
        self.mode = mode
        
sampler = MySampler(train_dataset, shuffle=False)
# list(MySampler([1, 2, 3, 10]))


# In[ ]:


repo_vecs = np.array([codebert_utils.code_to_vec(item) for item in train_inputs])
distances, matched_indices = nbrs.kneighbors(repo_vecs)
distance_priorities = 1 - (distances / distances.max()).squeeze()


# In[ ]:


sampler.set_priority(distance_priorities)
sampler.set_mode('active')


# In[ ]:


# distance_priorities
# sampler.set_curriculum(0.4)
# list(sampler)


# In[ ]:


# l[(np.sort(sampler.priority) < sampler.threshold).sum():]
# self.


# In[ ]:


from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)#, collate_fn=data_collator)
eval_dataloader = DataLoader(val_dataset, batch_size)#, collate_fn=data_collator)


# In[ ]:


# train_dataloader.sampler.set_priority(list(range(len(train_dataset) - 1)) + [-5])
# for ind, batch in train_dataloader:
#     print(ind)
# {k: v.shape for k, v in batch.items()}


# In[ ]:


# list(sampler)


# In[ ]:



now = datetime.now()
full_name = f'{name}_{exec_number}_{repo.rsplit("/", 1)[1][-20:]}_{sample_percent}'
# model_directory = f'{storage_directory}/tmp/finetuned/{full_name}'
# model_directory


# In[ ]:


tokenizer = T5Tokenizer.from_pretrained(load_model)
model = T5ForConditionalGeneration.from_pretrained(load_model)
model.resize_token_embeddings(len(tokenizer))
model.to('cuda')


# In[ ]:


num_train_epochs = 370


# In[ ]:


# predictions = np.array(outputs[1].argmax(-1).to('cpu'))
# labels = np.array(batch['labels'].to('cpu'))
# np.sum(np.all(np.equal(predictions, labels), axis=1))


# In[ ]:


from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)


# In[ ]:





# In[ ]:


from transformers import get_scheduler

num_training_steps = num_train_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=ws,
    num_training_steps=num_training_steps,
)
print(num_training_steps)


# In[ ]:


import copy


# In[ ]:


# for index, batch in train_dataloader:
#     batch = {k: v.to('cuda') for k, v in batch.items()}
#     print(index)#, batch)
#     break


# In[ ]:


# [value.item() for key, value in sorted(list(zip(list(sampler)[:16], outputs[1].max(-1).values.sum(1))))]


# In[ ]:


# model.eval()
# scores = []
# for batch in train_dataloader:
#     batch = {k: v.to('cuda') for k, v in batch.items()}
#     with torch.no_grad():
#         outputs = model(**batch)
#     scores += [item.item() for item in outputs[1].max(-1).values.mean(1).to('cpu')]
# scores    
# # # print(f'epoch #{epoch} | loss: {loss:.2f}, accuracy : {all_corrects/ all_cnt:.3f}')


# In[ ]:


from torch.nn.functional import softmax

# [item.item() for item in softmax(outputs[1], dim=-1).max(-1).values.prod(-1).to('cpu')]


# In[ ]:


# (outputs[1].argmax(-1) != 0).sum(-1) # length
# (batch['input_ids'] != 0).sum(1)


# In[ ]:


curriculum_list = [0.4, 0.65, 0.72, 0.85, 0.9, 0.95, 1, 1, 1, 1, 1,1 ,1 ,1 ,1 ,1 ,1 ,1, 1, 1, 1, 1]


# In[ ]:


from tqdm.auto import tqdm

progress_bar = tqdm(range(num_train_epochs * len(train_dataloader)))

model.train()
best_val_accuracy, best_val_loss = 0, 1
patience = 5
best_model = copy.deepcopy(model)
no_imp = 0

start_training = datetime.now()
curriculum = 0


if args.mode == 'distance_based':
    repo_vecs = np.array([codebert_utils.code_to_vec(item) for item in train_inputs])
    distances, matched_indices = nbrs.kneighbors(repo_vecs)
    distance_priorities = 1 - (distances / distances.max()).squeeze()
    sampler.set_priority(distance_priorities)
    sampler.set_mode('active')

for epoch in range(num_train_epochs):
#     if curriculum < 1:
#         curriculum += 0.2
#     sampler.curriculum = curriculum
    if epoch >= len(curriculum_list):
        sampler.curriculum = 1
    else:
        sampler.curriculum = curriculum_list[epoch]

    if args.mode != 'distance_based':
        sampler.set_mode('all')
        if epoch == 0  or True:
            model.eval()
            scores = []
            for batch in train_dataloader:
                batch = {k: v.to('cuda') for k, v in batch.items()}

                with torch.no_grad():
                    outputs = model(**batch)
                if args.mode == 'conf':
                    scores += [item.item() for item in softmax(outputs[1], dim=-1).max(-1).values.prod(-1).to('cpu')] #conf score
                elif args.mode == 'length_label':
                    scores += list((-1 * outputs[1].argmax(-1).to('cpu') != 0).sum(-1)) # length of generated labels
                elif args.mode == 'length_input':
                    scores += list((-1 * batch['input_ids'] != 0).sum(1).cpu()) # length of input
                elif args.mode == 'distance_based':
                    pass
                else:
                    raise Exception(f'Invalid argument args.mode: {args.mode}')
                # scores += [item.item() for item in outputs[1].max(-1).values.mean(1).to('cpu')]
            new_priorities = [value for key, value in sorted(list(zip(list(sampler), scores)))]
            sampler.set_priority(new_priorities)
            sampler.set_mode('active')
    #         print(len(sampler))
    #         print(list(sampler))
    
    model.train()
    all_corrects, all_cnt = 0, 0
    print('sampler', sampler.curriculum, len(sampler), len(train_dataloader))
    for batch in train_dataloader:
        batch = {k: v.to('cuda') for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs[0]#outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        
        predictions = np.array(outputs[1].argmax(-1).to('cpu'))
        labels = np.array(batch['labels'].to('cpu'))
        corrects = np.sum(np.all(np.equal(predictions, labels), axis=1))
        
        all_cnt += len(batch['labels'])
        all_corrects += corrects
        
#     print(f'epoch #{epoch} | loss: {loss:.2f}, accuracy : {all_corrects/ all_cnt:.3f}')    
    
    val_corrects, val_cnt = 0, 0
    for batch in eval_dataloader:
        batch = {k: v.to('cuda') for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        val_loss = outputs[0]
        predictions = np.array(outputs[1].argmax(-1).to('cpu'))
        labels = np.array(batch['labels'].to('cpu'))
        corrects = np.sum(np.all(np.equal(predictions, labels), axis=1))
        val_cnt += len(batch['labels'])
        val_corrects += corrects
        
    val_accuracy = val_corrects/ val_cnt
    print(f'epoch #{epoch} | tr_loss:{loss:.2f} tr_acc:{all_corrects/all_cnt} val_loss: {val_loss:.2f}, val_accuracy: {val_accuracy:.3f}')    
    print('---')
    
    
    if  val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        no_imp = 0
        best_model = copy.deepcopy(model)
        best_epoch = epoch
    else:
        no_imp += 1
    if no_imp >= patience:
        print(f'terminating... using {best_epoch}')
        break
    
    
    
end_training = datetime.now()

# In[ ]:


all_corrects, all_cnt = 0, 0
best_model.eval()
for batch in eval_dataloader:
    batch = {k: v.to('cuda') for k, v in batch.items()}
    with torch.no_grad():
        outputs = best_model(**batch)
    loss = outputs[0]
    predictions = np.array(outputs[1].argmax(-1).to('cpu'))
    labels = np.array(batch['labels'].to('cpu'))
    corrects = np.sum(np.all(np.equal(predictions, labels), axis=1))

    all_cnt += len(batch['labels'])
    all_corrects += corrects
    
print(f'epoch #{epoch} | loss: {loss:.2f}, accuracy : {all_corrects/ all_cnt:.3f}')


# In[ ]:


len(outputs)


# In[ ]:


sampler.shuffle


# In[ ]:


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


# In[ ]:



training_args = Seq2SeqTrainingArguments(
    output_dir=model_address + '/tr',
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=ws,
    weight_decay=wd,
    logging_dir=model_address + '/tr',
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

from transformers import EarlyStoppingCallback


trainer = Seq2SeqTrainer(
    model=best_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=[torch.optim.Adam(params=model.parameters(), lr=lr), None],
    tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
)


# In[ ]:


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


# In[ ]:





# In[ ]:


# from transformers import EarlyStoppingCallback

# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     optimizers=[torch.optim.Adam(params=model.parameters(), lr=lr), None],
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
# )


# In[ ]:




# trainer.train()



# In[ ]:



# tuned_model_dir = f'{model_directory}/best'
# tuned_model_dir='/scratch/arminz/tmp/finetuned'
trainer.save_model(model_address)


end_all = datetime.now()
import csv
with open('tuner_runtime.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([name, repo, len(train_dataset), len(val_dataset), base_model, start_all, start_training, end_training, end_all])

# In[78]:


# In[ ]:


if local:
    from numba import cuda
    device = cuda.get_current_device()
    device.reset()
#


# In[ ]:


# result = os.system(f'python hf_transformers/tfix_testing.py --load-model {tuned_model_dir} -bs 16 --model-name t5-large -d repo-based-included -r {repo}')
# print(result)
#
# result = os.system(f'python hf_transformers/tfix_testing.py --load-model {tuned_model_dir} -bs 16 --model-name t5-large -d source-test')
# print(result)
#
# # In[45]:
#
#
# import shutil
#
# shutil.rmtree(tuned_model_dir)
#


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




