from collections import defaultdict
from datetime import datetime
import argparse
import json
import os
import sys
from typing import DefaultDict, List

sys.path.append(".")

from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from transformers import set_seed
import numpy as np
import torch

from data_reader import DataPoint, GetDataAsPython
from prepare_data import create_data
from prepare_data import create_dataset
from prepare_data import extract_warning_types
from prepare_data import filter_rule
from utils import boolean_string
from utils import get_scores_weighted_average
from utils import get_current_time

# transformers.logging.set_verbosity_info()
set_seed(42)
print("start time: ", get_current_time())


parser = argparse.ArgumentParser()
parser.add_argument("-bs", "--batch-size", type=int, default=32)
parser.add_argument("-mn", "--model-name", type=str, choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"], required=True,)
parser.add_argument("-lm", "--load-model", type=str, default="")  # Checkpoint dir to load the model. Example: t5-small_global_14-12-2020_16-29-22/checkpoint-10
parser.add_argument("-ea", "--eval-all", type=boolean_string, default=False)  # to evaluate on all data or not
parser.add_argument("-eas", "--eval-acc-steps", type=int, default=1)
parser.add_argument("-md", "--result-dir", type=str, default="")
parser.add_argument("-et", "--error-type", type=str, default="")
parser.add_argument("-d", "--design", type=str, required=True, choices=['old', 'new', 'repo-based-included'])
parser.add_argument("-r", "--repo", type=str, required=False)
args = parser.parse_args()

local = False

model_name = args.model_name

if local:
    storage_directory = './storage/'
else:
    storage_directory = '/scratch/arminz/'

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
# Create job's directory
if args.result_dir != "":
    test_result_directory = args.result_dir
else:
    if not args.repo:
        test_result_directory = f'{storage_directory}/{now.day}/{model_name}_test_{args.design}_{dt_string}'
    else:
        test_result_directory = f'{storage_directory}/{now.day}/per-repo/{model_name}_test_{args.repo.rsplit("/", 1)[1][-20:]}_{dt_string}'

os.makedirs(test_result_directory, exist_ok=True)
with open(os.path.join(test_result_directory, "commandline_args.txt"), "w") as f:
    f.write("\n".join(sys.argv[1:]))

# Read data
data = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_repo_specific_final.json")
data_eslint = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_eslint_final.json")

data += data_eslint
all_warning_types = extract_warning_types(data)
if args.error_type != "":
    all_warning_types = [args.error_type]
print(all_warning_types)
(train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels, train_info, val_info, test_info, ) =\
    create_data(data, all_warning_types, include_warning=True, design=args.design, select_repo=args.repo)

# Load the tokenizer and the model that will be tested.
tokenizer = T5Tokenizer.from_pretrained(args.load_model)
print("Loaded tokenizer from directory {}".format(args.load_model))
model = T5ForConditionalGeneration.from_pretrained(args.load_model)
print("Loaded model from directory {}".format(args.load_model))
print(f"cuda:{torch.cuda.current_device()}")
model.to(f"cuda:{torch.cuda.current_device()}")
model.resize_token_embeddings(len(tokenizer))
model.eval()

# Create dataset required by pytorch
train_dataset = create_dataset(
    train_inputs, train_labels, tokenizer, pad_truncate=True, max_length=128
)
val_dataset = create_dataset(val_inputs, val_labels, tokenizer, pad_truncate=True)

# Trainer arguments.
# Note that Seq2SeqTrainer class has a method predict() that will be used to generate predictions.
# That is why we still need to create a trainer instance and its arguments even though we are in testing
training_args = Seq2SeqTrainingArguments(
    output_dir=test_result_directory,
    num_train_epochs=0,
    per_device_eval_batch_size=args.batch_size,
    logging_dir=test_result_directory,
    logging_steps=100,
    do_eval=True,
    evaluation_strategy="epoch",
    eval_accumulation_steps=args.eval_acc_steps,  # set this lower, if testing or validation crashes
    predict_with_generate=True,  # never set this to false, it is for testing.
    seed=42,  # default value
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

print("Testing has started")
if args.eval_all:
    # add random_test samples to the testing data
    counter = 0
    for key in test_inputs:
        counter += len(test_inputs[key])
    print("size before addition of filtered samples: ", counter)

    file_paths = [
        f"{storage_directory}/data_and_models/data/data_autofix_tracking_repo_specific_filtered.json",
        f"{storage_directory}/data_and_models/data/data_autofix_tracking_eslint_filtered.json",
    ]

    for path in file_paths:
        filtered_data = GetDataAsPython(path)
        print("size of filtered data: ", len(filtered_data))

        filtered_test: DefaultDict[str, List[str]] = defaultdict(list)
        filtered_test_labels: DefaultDict[str, List[str]] = defaultdict(list)
        filtered_test_info: DefaultDict[str, List[DataPoint]] = defaultdict(list)

        filtered_warning_types = extract_warning_types(filtered_data)
        if args.error_type != "":
            filtered_warning_types = [args.error_type]

        for warning in filtered_warning_types:
            filtered_test_data = filter_rule(filtered_data, warning)

            filtered_test_inputs = [
                data_point.GetT5Representation(True)[0] for data_point in filtered_test_data
            ]
            filtered_test_outputs = [
                data_point.GetT5Representation(True)[1] for data_point in filtered_test_data
            ]

            filtered_test[warning] = filtered_test_inputs
            filtered_test_labels[warning] = filtered_test_outputs

            filtered_test_info[warning] = filtered_test_data

        for warning in filtered_warning_types:
            test_inputs[warning] = test_inputs[warning] + filtered_test[warning]
            test_labels[warning] = test_labels[warning] + filtered_test_labels[warning]
            test_info[warning] = test_info[warning] + filtered_test_info[warning]

counter = 0
for key in test_inputs:
    counter += len(test_inputs[key])
print("Number of testing samples: ", counter)

# test that the samples are well aligned among inputs and info
for warning in test_inputs:
    inputs = test_inputs[warning]
    infos = test_info[warning]
    for i, code in enumerate(inputs):
        assert code == infos[i].GetT5Representation(True)[0], "something wrong! stop it!"

# Generate predictions
scores: DefaultDict[str, float] = defaultdict(float)
counts: DefaultDict[str, float] = defaultdict(int)
for i, warning in enumerate(all_warning_types):
    test_warning = test_inputs[warning]
    test_warning_labels = test_labels[warning]
    test_warning_info = test_info[warning]
    target_max_length = 256  # Set this to 256 if enough memory
    if not test_warning:
        scores[warning] = 'NA'
        counts[warning] = 0
        continue
    # print(f"rule {i}: {warning}, # {len(test_warning)}")
    test_warning_dataset = create_dataset(
        test_warning,
        test_warning_labels,
        tokenizer,
        pad_truncate=True,
        max_length=target_max_length,
    )

    target_ids = tokenizer(
        test_warning_labels,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=target_max_length,
    ).input_ids
    target_ids = np.array(target_ids)

    output_ids = trainer.predict(
        test_dataset=test_warning_dataset, num_beams=5, max_length=target_max_length
    ).predictions
    output_ids = np.pad(
        output_ids, ((0, 0), (0, target_max_length - output_ids.shape[1])), mode="constant"
    )
    output_ids = np.delete(output_ids, 0, axis=1)
    output_ids = np.insert(output_ids, target_max_length - 1, 0, axis=1)

    correct_counter = np.sum(np.all(np.equal(target_ids, output_ids), axis=1))
    total_counter = len(output_ids)
    for k, output_id in enumerate(output_ids):
        pred = tokenizer.decode(output_id, skip_special_tokens=True)
        predictions = []
        predictions.append(pred)
        test_warning_info[k].predictions = predictions

    scores[warning] = correct_counter / total_counter
    counts[warning] = total_counter
    test_info[warning] = test_warning_info
    # print(f"rule {i} acc: {correct_counter / total_counter}")

average, count = get_scores_weighted_average(scores, counts)
number_of_warnings = len([scores[k] for k in scores if scores[k] != 'NA'])

assert count == counter, 'counts must be equal'

scores["average"] = average
scores['number_of_warnings'] = number_of_warnings
scores['samples_count'] = counter

print(f'score average {average}')

if args.repo:
    with open(f'{storage_directory}/results_per_repo.csv', 'a') as f:
        f.write(f'{args.repo},{scores["average"]:.2f},{scores["number_of_warnings"]},{scores["samples_count"]},{dt_string},{model_name},{args.load_model}\n')
# create the whole test list
test_list: List[DataPoint] = []
for key in test_info:
    test_list += test_info[key]
print(f'result : {test_result_directory}')
with open(os.path.join(test_result_directory, "test_data.json"), "w") as outfile:
    json.dump(
        [datapoint.__dict__ for datapoint in test_list], outfile, default=lambda o: o.__dict__
    )

serialized_scores = json.dumps(scores, indent=4)
output_file = open(os.path.join(test_result_directory, "first_accs.txt"), "w+")
output_file.write(serialized_scores)
output_file.close()

print("end time: ", get_current_time())
