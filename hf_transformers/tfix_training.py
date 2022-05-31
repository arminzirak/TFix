from datetime import datetime
import argparse
import os
import sys

sys.path.append(".")

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
from datetime import datetime

# transformers.logging.set_verbosity_info()
set_seed(42)
print("start time: ", get_current_time())

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=30)
parser.add_argument("-bs", "--batch-size", type=int, default=32)
parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4)
parser.add_argument("-gcv", "--gradient-clip-val", type=float, default=0.0)
parser.add_argument("-wd", "--weight-decay", type=float, default=0)
parser.add_argument("-mn", "--model-name", type=str, choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"], required=True)
parser.add_argument("-eas", "--eval-acc-steps", type=int, default=1)
parser.add_argument("-md", "--model-dir", type=str, default="")
parser.add_argument("-et", "--error-type", type=str, default="")
parser.add_argument("-stl", "--save-total-limit", type=int, default=-1)
parser.add_argument("-pt", "--pre-trained", type=boolean_string, default=True)
parser.add_argument("-d", "--design", type=str, required=True, choices=['old', 'new', 'repo-based', 'repo-based-included'])
args = parser.parse_args()

start_all = datetime.now()

import socket
local = False if 'computecanada' in socket.gethostname() else True

model_name = args.model_name
name = 'train'
if local:
    storage_directory = './storage/'
    pretrained_model = model_name
else:
    storage_directory = '/scratch/arminz/'
    pretrained_model = f'{storage_directory}/pretrained/{model_name}'


# Create job directory
if args.model_dir != "":
    model_directory = args.model_dir
else:
    raise('this format is not handled')
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    # model_directory = "t5global" + "_" + dt_string
    model_directory = f'{storage_directory}/training/{model_name}_global_{args.design}_{dt_string}'
print(f'model dir: {model_directory}')

os.makedirs(model_directory, exist_ok=True)
with open(os.path.join(model_directory, "commandline_args.txt"), "w") as f:
    f.write("\n".join(sys.argv[1:]))

# Read and prepare data
data = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_repo_specific_final.json")
data_eslint = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_eslint_final.json")
data += data_eslint
all_warning_types = extract_warning_types(data)
if args.error_type != "":
    all_warning_types = [args.error_type]
print(all_warning_types)
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
) = create_data(data, all_warning_types, include_warning=True, design=args.design)

# Create the tokenizer and the model
tokenizer = T5Tokenizer.from_pretrained(
    pretrained_model,
)
tokenizer.add_tokens(["{", "}", ">", "\\", "^"])
tokenizer.save_pretrained(model_directory)
if args.pre_trained:
    model = T5ForConditionalGeneration.from_pretrained(pretrained_model, return_dict=False)
else:
    print("Training from scratch")
    config = T5Config.from_pretrained(pretrained_model)
    model = T5ForConditionalGeneration(config)
model.parallelize()
model.resize_token_embeddings(len(tokenizer))
print("Models parameters: ", model.num_parameters())

# Create dataset required by pytorch
print(len(train_inputs), len(train_labels))
train_dataset = create_dataset(
    train_inputs, train_labels, tokenizer, pad_truncate=True, max_length=128
)
val_dataset = create_dataset(val_inputs, val_labels, tokenizer, pad_truncate=True)
# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=model_directory,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    warmup_steps=500,
    weight_decay=args.weight_decay,
    logging_dir=model_directory,
    logging_steps=100,
    do_eval=True,
    evaluation_strategy="epoch",
    learning_rate=args.learning_rate,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=args.epochs if args.save_total_limit == -1 else args.save_total_limit,
    eval_accumulation_steps=args.eval_acc_steps,  # set this lower, if testing or validation crashes
    disable_tqdm=False,
    predict_with_generate=True,  # never set this to false.
    seed=42,  # default value
)
# Create trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=[torch.optim.Adam(params=model.parameters(), lr=args.learning_rate), None],
    tokenizer=tokenizer,
)
start_training = datetime.now()

print("training start time: ", get_current_time())
trainer.train()
print("end time: ", get_current_time())

end_training = datetime.now()
end_all = datetime.now()
import csv
with open('tuner_runtime.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([name, 'all', args.design, args.epochs, model_directory, len(train_dataset), len(val_dataset), start_all, start_training, end_training, end_all])