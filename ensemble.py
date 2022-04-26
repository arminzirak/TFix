#!/usr/bin/env python
# coding: utf-8

# In[212]:


import sys
sys.path.append('.')
sys.path.append('./hf_transformers')


# In[213]:


from collections import defaultdict
from datetime import datetime
import argparse
import json
import os
from typing import DefaultDict, List


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


# In[214]:


repo = '/data/all/data/elastic/kibana'
model_name = 't5-small'
tuned_model_address = './storage/tmp/finetuned/good_333_kibana_1.0/best'
general_model_address = './storage/training/t5-small_repo-based_21-01-2022_10-29-42/checkpoint-16440'


# In[215]:
repo_list = ['/data/all/data/emberjs/ember.js', '/data/all/data/Encapsule-Annex/onm', '/data/all/data/sequelize/sequelize',
             '/data/all/data/dcos/dcos-ui', '/data/all/data/LivelyKernel/LivelyKernel', '/data/all/data/svgdotjs/svg.js', '/data/all/data/foam-framework/foam']

tuned_model_address_list = ['./storage/tmp/finetuned/good_806_ember.js_1.0/best', './storage/tmp/finetuned/good_861_onm_1.0/best',
                            './storage/tmp/finetuned/good_446_sequelize_1.0/best', './storage/tmp/finetuned/good_145_dcos-ui_1.0/best', './storage/tmp/finetuned/good_132_LivelyKernel_1.0/best',
                            './storage/tmp/finetuned/good_561_svg.js_1.0/best', './storage/tmp/finetuned/good_314_foam_1.0/best']
device = 'cuda'

# In[277]:
# repo_list = ['/data/all/data/qooxdoo/qooxdoo']
# tuned_model_address_list = ['./storage/tmp/finetuned/good_978_qooxdoo_1.0/best']

# Load the tokenizer and the model that will be tested.
general_tokenizer = T5Tokenizer.from_pretrained(f'{general_model_address}')
print("Loaded tokenizer from directory {}".format(f'{general_model_address}'))
general_model = T5ForConditionalGeneration.from_pretrained(f'{general_model_address}')
print("Loaded model from directory {}".format(f'{general_model_address}'))
print(f"cuda:{torch.cuda.current_device()}")
# general_model.to(f"cuda:{torch.cuda.current_device()}")
general_model.resize_token_embeddings(len(general_tokenizer))
general_model.eval()
general_model.to(device)
general_model.device

# In[278]:


for repo, tuned_model_address in list(zip(repo_list, tuned_model_address_list))[5:]:
    print(repo, tuned_model_address)


    # Load the tokenizer and the model that will be tested.
    tuned_tokenizer = T5Tokenizer.from_pretrained(f'{tuned_model_address}')
    print("Loaded tokenizer from directory {}".format(f'{tuned_model_address}'))
    tuned_model = T5ForConditionalGeneration.from_pretrained(f'{tuned_model_address}')
    print("Loaded model from directory {}".format(f'{tuned_model_address}'))
    print(f"cuda:{torch.cuda.current_device()}")
    # tuned_model.to(f"cuda:{torch.cuda.current_device()}")
    tuned_model.resize_token_embeddings(len(general_tokenizer))
    tuned_model.eval()
    tuned_model.to(device)
    tuned_model.device

    # In[272]:


    COEFF_LIST = np.unique([0] + [int(1.3 ** i) for i in range(30)])


    for coeff in COEFF_LIST:

        # In[273]:


        # parser = argparse.ArgumentParser()
        # parser.add_argument("-bs", "--batch-size", type=int, default=32)
        # parser.add_argument("-mn", "--model-name", type=str, choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"], required=True,)
        # parser.add_argument("-lm", "--load-model", type=str, default="")  # Checkpoint dir to load the model. Example: t5-small_global_14-12-2020_16-29-22/checkpoint-10
        # parser.add_argument("-ea", "--eval-all", type=boolean_string, default=False)  # to evaluate on all data or not
        # parser.add_argument("-eas", "--eval-acc-steps", type=int, default=1)
        # parser.add_argument("-md", "--result-dir", type=str, default="")
        # parser.add_argument("-et", "--error-type", type=str, default="")
        # parser.add_argument("-d", "--design", type=str, required=True, choices=['old', 'new', 'repo-based-included'])
        # parser.add_argument("-r", "--repo", type=str, required=False)
        # args = parser.parse_args()


        # In[274]:



        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        # Create job's directory

        test_result_directory = f'./test'
        storage_directory = f'./storage'

        os.makedirs(test_result_directory, exist_ok=True)
        with open(os.path.join(test_result_directory, "commandline_args.txt"), "w") as f:
            f.write("\n".join(sys.argv[1:]))

        # Read data
        data = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_repo_specific_final.json")
        data_eslint = GetDataAsPython(f"{storage_directory}/data_and_models/data/data_autofix_tracking_eslint_final.json")
        data += data_eslint


        # In[275]:


        len(data)


        # In[276]:


        all_warning_types = extract_warning_types(data)
        # if args.error_type != "":
        #     all_warning_types = [args.error_type]
        print(all_warning_types)
        (train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels, train_info, val_info, test_info, ) =    create_data(data, all_warning_types, include_warning=True, design='repo-based-included', select_repo=repo)


        # In[ ]:







        # In[279]:



        # Create dataset required by pytorch
        # general_train_dataset = create_dataset(
        #     train_inputs, train_labels, general_tokenizer, pad_truncate=True, max_length=128
        # )
        # general_val_dataset = create_dataset(val_inputs, val_labels, general_tokenizer, pad_truncate=True)

        # # Trainer arguments.
        # # Note that Seq2SeqTrainer class has a method predict() that will be used to generate predictions.
        # # That is why we still need to create a trainer instance and its arguments even though we are in testing
        # training_args = Seq2SeqTrainingArguments(
        #     output_dir=test_result_directory,
        #     num_train_epochs=0,
        #     per_device_eval_batch_size=16,
        #     logging_dir=test_result_directory,
        #     logging_steps=100,
        #     do_eval=True,
        #     evaluation_strategy="epoch",
        #     eval_accumulation_steps=1,  # set this lower, if testing or validation crashes
        #     predict_with_generate=True,  # never set this to false, it is for testing.
        #     seed=42,  # default value
        # )

        # general_trainer = Seq2SeqTrainer(
        #     model=general_model,
        #     args=training_args,
        #     train_dataset=general_train_dataset,
        #     eval_dataset=general_val_dataset,
        #     tokenizer=general_tokenizer,
        # )


        # In[280]:


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


        # In[ ]:


        # Generate predictions
        target_max_length = 256  # Set this to 256 if enough memory
        import gc
        scores: DefaultDict[str, float] = defaultdict(float)
        counts: DefaultDict[str, float] = defaultdict(int)
        for i, warning in enumerate(all_warning_types):

            test_warning = test_inputs[warning]
            test_warning_labels = test_labels[warning]
            test_warning_info = test_info[warning]

            if not test_warning:
                scores[warning] = 'NA'
                counts[warning] = 0
                continue
        #     print('coding general')

            train_ids = general_tokenizer(
                test_warning,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=target_max_length,
                ).input_ids
            predictions = general_model.generate(train_ids.to(device), max_length=target_max_length, output_scores=True, num_return_sequences=5, num_beams=5, return_dict_in_generate=True)
            output_ids = np.pad(
                predictions.sequences.cpu(), ((0, 0), (0, target_max_length - predictions.sequences.shape[1])), mode="constant"
            )
            prediction_scores = predictions.sequences_scores.cpu()
            del predictions
            gc.collect()
        #     print('coding tuned')
            # print(target_ids.shape)
            train_ids_t = tuned_tokenizer(
                test_warning,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=target_max_length,
            ).input_ids
            predictions_tuned = tuned_model.generate(train_ids_t.to(device), max_length=target_max_length, output_scores=True, num_return_sequences=5, num_beams=5, return_dict_in_generate=True)
            output_ids_tuned = np.pad(
                predictions_tuned.sequences.cpu(), ((0, 0), (0, target_max_length - predictions_tuned.sequences.shape[1])), mode="constant"
            )
            prediction_scores_tuned = predictions_tuned.sequences_scores.cpu()

            del predictions_tuned
        #     print(prediction_scores, prediction_scores_tuned)
            assert len(output_ids) == 5 * len(test_warning)
            all_predictions = []
            for j in range(len(test_warning)):
                predictions_aggregate = defaultdict(int)
                for prediction, score in zip(output_ids[j * 5: (j + 1) * 5], prediction_scores[j * 5: (j + 1) * 5]):
                    predictions_aggregate[','.join([str(item.item()) for item in prediction])] += score.item() + 1
        #             print(prediction, score)

                for prediction_tuned, score in zip(output_ids_tuned[j * 5: (j + 1) * 5], prediction_scores_tuned[j * 5: (j + 1) * 5]):
                    predictions_aggregate[','.join([str(item.item()) for item in prediction_tuned])] += coeff * (score.item() + 1)
            #         print(prediction, score)
                picked_result = [int(item) for item in max(predictions_aggregate, key=predictions_aggregate.get).split(',')]
                all_predictions.append(picked_result)
            all_predictions = np.array(all_predictions, dtype=int)

        #     target_max_length = 256  # Set this to 256 if enough memory
        #     if not test_warning:
        #         scores[warning] = 'NA'
        #         counts[warning] = 0
        #         continue
        #     # print(f"rule {i}: {warning}, # {len(test_warning)}")
        #     test_warning_dataset = create_dataset(
        #         test_warning,
        #         test_warning_labels,
        #         general_tokenizer,
        #         pad_truncate=True,
        #         max_length=target_max_length,
        #     )
        #     print('decoding')
            target_ids = general_tokenizer(
                test_warning_labels,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=target_max_length,
            ).input_ids
            target_ids = np.array(target_ids)

        #     output_ids = general_trainer.predict(
        #         test_dataset=test_warning_dataset, num_beams=5, max_length=target_max_length
        #     ).predictions
        #     output_ids = np.pad(
        #         all_predictions, ((0, 0), (0, target_max_length - all_predictions.shape[1])), mode="constant"
        #     )
            output_ids = np.delete(all_predictions, 0, axis=1)
            output_ids = np.insert(output_ids, target_max_length - 1, 0, axis=1)

            correct_counter = np.sum(np.all(np.equal(target_ids, output_ids), axis=1))
            total_counter = len(output_ids)
            for k, output_id in enumerate(output_ids):
                pred = general_tokenizer.decode(output_id, skip_special_tokens=True)
                predictions = []
                predictions.append(pred)
                test_warning_info[k].predictions = predictions

            scores[warning] = correct_counter / total_counter
            counts[warning] = total_counter
            test_info[warning] = test_warning_info
            print(f"rule {i} acc: {correct_counter / total_counter}")
            predictions_tuned, predictions, output_ids, output_ids_tuned,predictions_aggregate, target_ids, train_ids_t, train_ids = [], [],[],[],[],[],[],[]
            pred = []
            gc.collect()

        #     break


        # In[ ]:


        # for prediction_tuned, score in zip(output_ids_tuned[j * 3: (j + 1) * 3], prediction_scores_tuned[i * 3: (i + 1) * 3]):
        #     predictions_aggregate[','.join([str(item.item()) for item in prediction_tuned])] += 1.1 * (score.item() + 0.2)
        #     print('f')


        # In[ ]:





        # In[ ]:


        # (train_ids_t == train_ids).sum()/(48 * 256)


        # In[ ]:



        average, count = get_scores_weighted_average(scores, counts)
        number_of_warnings = len([scores[k] for k in scores if scores[k] != 'NA'])

        assert count == counter, 'counts must be equal'

        scores["average"] = average
        scores['number_of_warnings'] = number_of_warnings
        scores['samples_count'] = counter

        print(f'score average: {average} samples_count: {scores["samples_count"]}')


        # In[ ]:


        with open(f'{storage_directory}/results.csv', 'a') as f:
            f.write(f'ensembleA,{repo if repo else "all"},{scores["average"]:.2f},{scores["number_of_warnings"]},{scores["samples_count"]},{dt_string},{model_name},{tuned_model_address},{coeff}\n')


        # In[ ]:


        # output_ids.shape


        # In[ ]:


        # print(target_ids.shape)
        # train_ids = general_tokenizer(
        #     test_warning,
        #     return_tensors="pt",
        #     truncation=True,
        #     padding="max_length",
        #     max_length=target_max_length,
        # ).input_ids
        # predictions = general_model.generate(train_ids.to('cuda'), max_length=target_max_length, output_scores=True, num_return_sequences=3, num_beams=5, return_dict_in_generate=True)
        # prediction_ids = prediction.sequences.cpu()


        # In[ ]:


        # # print(target_ids.shape)
        # train_ids_t = tuned_tokenizer(
        #     test_warning,
        #     return_tensors="pt",
        #     truncation=True,
        #     padding="max_length",
        #     max_length=target_max_length,
        # ).input_ids
        # predictions_tuned = tuned_model.generate(train_ids.to('cuda'), max_length=target_max_length, output_scores=True, num_return_sequences=3, num_beams=5, return_dict_in_generate=True)
        # prediction_ids = prediction.sequences.cpu()


        # In[ ]:


        # (train_ids_t == train_ids).all(), (predictions.sequences == predictions_tuned.sequences).all()


        # In[ ]:


        # predictions.sequences_scores


        # In[ ]:


        # for i in range(len(test_warning)):
        #     predictions_aggregate = defaultdict(int)
        #     for prediction, score in zip(predictions.sequences[i * 3: (i + 1) * 3], predictions.sequences_scores[i * 3: (i + 1) * 3]):
        #         predictions_aggregate[','.join([str(item.item()) for item in prediction])] += score.item() + 0.2
        # #         print(prediction, score)

        #     for prediction_tuned, score in zip(predictions_tuned.sequences[i * 3: (i + 1) * 3], predictions_tuned.sequences_scores[i * 3: (i + 1) * 3]):
        #         predictions_aggregate[','.join([str(item.item()) for item in prediction_tuned])] += 1.1 * (score.item() + 0.2)
        # #         print(prediction, score)

        #     break


        # In[ ]:


        # picked_result = max(predictions_aggregate, key=predictions_aggregate.get).split(',')
        # picked_result


        # In[ ]:


        # predictions.sequences_scores[0:3], predictions_tuned.sequences_scores[0:3]


        # In[ ]:


        # prediction_ids = np.pad(
        #     prediction_ids, ((0, 0), (0, target_max_length - prediction_ids.shape[1])), mode="constant"
        # )
        # prediction_ids = np.delete(prediction_ids, 0, axis=1)
        # prediction_ids = np.insert(prediction_ids, target_max_length - 1, 0, axis=1)
        # o_prediction = general_trainer.predict(test_dataset=test_warning_dataset, num_beams=5, max_length=target_max_length)
        # output_ids = o_prediction.predictions
        # output_ids = np.pad(
        #     output_ids, ((0, 0), (0, target_max_length - output_ids.shape[1])), mode="constant"
        # )
        # output_ids = np.delete(output_ids, 0, axis=1)
        # output_ids = np.insert(output_ids, target_max_length - 1, 0, axis=1)
        # (output_ids == prediction_ids).all()


        # In[ ]:


        # prediction_ids = np.pad(
        #     prediction_ids, ((0, 0), (0, target_max_length - prediction_ids.shape[1])), mode="constant"
        # )
        # prediction_ids = np.delete(prediction_ids, 0, axis=1)
        # prediction_ids = np.insert(prediction_ids, target_max_length - 1, 0, axis=1)


        # In[ ]:


        # general_model.generate(train_ids.to('cuda'), max_length=target_max_length, output_scores=True, num_return_sequences=1, num_beams=5).sequences


        # In[ ]:


        # correct_counter = np.sum(np.all(np.equal(target_ids, output_ids), axis=1))
        # total_counter = len(output_ids)
        # for k, output_id in enumerate(output_ids):
        #     pred = general_tokenizer.decode(output_id, skip_special_tokens=True)
        #     predictions = []
        #     predictions.append(pred)
        #     test_warning_info[k].predictions = predictions


        # In[270]:


        # scores[warning] = correct_counter / total_counter
        # counts[warning] = total_counter
        # test_info[warning] = test_warning_info


        # In[271]:


        # prediction.predictions.shape


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

    del tuned_model, tuned_tokenizer


    # In[ ]:




