#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime
import argparse
import os
import sys

sys.path.append(".")
sys.path.append("./hf_transformers")

from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import T5Config
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from transformers import set_seed
import torch

from data_reader import GetDataAsPython
from prepare_data import create_data
from prepare_data import create_dataset
from prepare_data import extract_warning_types
from utils import boolean_string
from utils import get_current_time


# In[2]:


set_seed(42)


# In[3]:


import socket
local = False if 'computecanada' in socket.gethostname() else True

large = False


# In[4]:


model_name = 't5-small' if not large else 't5-large'


# In[5]:


if local:
    assert not large, 'large cannot be trained on local'
    storage_directory = './storage/'
    pretrained_model = model_name
else:
    storage_directory = '/scratch/arminz/'
    pretrained_model = f'{storage_directory}/pretrained/{model_name}'


# In[6]:


model_dir = "" # args.model_dir
design = "repo-based" #args.design
pre_trained = True #args.pre_trained
epochs = 30 #args.epochs
if not large:
    batch_size = 16 if local else 64
else:
    assert not local
    batch_size = 16
save_total_limit = 1 # args.save-total-limit
eval_acc_steps = 1 # eval-acc-steps
learning_rate = 1e-4 # args.learning-rate
weight_decay = 0 # args.weight-decay


# In[7]:


# Create job directory
if model_dir != "":
    model_directory = args.model_dir
else:
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    # model_directory = "t5global" + "_" + dt_string
    model_directory = f'{storage_directory}/training-tbug/{model_name}_{design}_{dt_string}'


# In[8]:


# os.makedirs(model_directory)
# with open(os.path.join(model_directory, "commandline_args.txt"), "w") as f:
#     f.write("\n".join(sys.argv[1:]))

# Read and prepare data
data = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_repo_specific_final.json")
data_eslint = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_eslint_final.json")
data += data_eslint
all_warning_types = extract_warning_types(data)
# if args.error_type != "":
#     all_warning_types = [args.error_type]
print(all_warning_types)


# In[9]:


(
    train_inputs,
    train_labels,
    val_inputs,
    val_labels,
    test_inputs,
    test_labels,
    train_info,
    val_info,
    test_info,
) = create_data(data, all_warning_types, include_warning=True, design=design, back_translation=True)


# In[10]:


# Create the tokenizer and the model
tokenizer = T5Tokenizer.from_pretrained(
    pretrained_model,
)
tokenizer.add_tokens(["{", "}", ">", "\\", "^"])
tokenizer.save_pretrained(model_directory)
if pre_trained:
    model = T5ForConditionalGeneration.from_pretrained(pretrained_model, return_dict=False)
else:
    print("Training from scratch")
    config = T5Config.from_pretrained(pretrained_model)
    model = T5ForConditionalGeneration(config)
model.parallelize()
model.resize_token_embeddings(len(tokenizer))
print("Models parameters: ", model.num_parameters())


# In[11]:


model


# In[12]:


print(len(train_inputs), len(train_labels))
train_dataset = create_dataset(
    train_labels, train_inputs, tokenizer, pad_truncate=True, max_length=128
)
val_dataset = create_dataset(val_labels, val_inputs, tokenizer, pad_truncate=True)


# In[13]:


train_labels[10]


# In[ ]:





# In[14]:


# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=model_directory,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=500,
    weight_decay=weight_decay,
    logging_dir=model_directory,
    logging_steps=100,
    do_eval=True,
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=args.epochs if save_total_limit == -1 else save_total_limit,
    eval_accumulation_steps=eval_acc_steps,  # set this lower, if testing or validation crashes
    disable_tqdm=False,
    predict_with_generate=True,  # never set this to false.
    seed=42,  # default value
)


# In[16]:


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=[torch.optim.Adam(params=model.parameters(), lr=learning_rate), None],
    tokenizer=tokenizer,
)


# In[ ]:


trainer.train()


# In[ ]:


model.device


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




