from collections import defaultdict
from typing import Any, DefaultDict, List, Dict

from transformers.tokenization_utils import PreTrainedTokenizer

from sklearn.model_selection import train_test_split
import torch
from transformers import BatchEncoding

from data_reader import DataPoint, MinimalDataPoint
from collections import Counter
import pandas as pd
import random


def extract_warning_types(data: List[DataPoint]) -> List[str]:
    all_warnings: List[str] = []
    for sample in data:
        if sample.linter_report.rule_id not in all_warnings:
            all_warnings.append(sample.linter_report.rule_id)
    return all_warnings


def filter_rule(data: List[DataPoint], rule_type: str) -> List[DataPoint]:
    filtered_data: List[DataPoint] = []
    for point in data:
        if isinstance(point, MinimalDataPoint):
            if point.t5_representation[len('fix '): len('fix ') + len(rule_type)] == rule_type:
                filtered_data.append(point)
        else:
            if point.linter_report.rule_id == rule_type:
                filtered_data.append(point)
    return filtered_data


def split_filtered(filtered_data: List[DataPoint], include_warning: bool, design: str, select_repo=None,
                   back_translation=False, seed=13, no_split=False, no_valid=False):
    filtered_data_temp = filtered_data

    if not back_translation:
        inputs = [data_point.GetT5Representation(include_warning)[0] for data_point in filtered_data]
        outputs = [data_point.GetT5Representation(include_warning)[1] for data_point in filtered_data_temp]
    else:
        inputs = [data_point.GetTBugRepresentation(include_warning)[0] for data_point in filtered_data]
        outputs = [data_point.GetTBugRepresentation(include_warning)[1] for data_point in filtered_data_temp]

    if design == 'new':
        raise Exception('deprecated design')
        # repos = pd.read_csv('./repos.csv', index_col=0)
        # target_repos = repos[repos['category'] == 'target']
        # train_inputs, train_labels, train_info = list(), list(), list()
        # test_inputs, test_labels, test_info = list(), list(), list()
        #
        # for input_instance, output_instance, filtered_data_instance in zip(inputs, outputs, filtered_data):
        #     if not (target_repos['repo'] == filtered_data_instance.repo).any():
        #         train_inputs.append(input_instance)
        #         train_labels.append(output_instance)
        #         train_info.append(filtered_data_instance)
        #     elif (not select_repo) or (select_repo == filtered_data_instance.repo):
        #         test_inputs.append(input_instance)
        #         test_labels.append(output_instance)
        #         test_info.append(filtered_data_instance)

    elif design == 'old':
        raise Exception('deprecated design')
        # test_size = 0.1 if len(inputs) >= 10 else 1 / len(inputs)
        #
        # train_inputs, test_inputs, train_labels, test_labels = train_test_split(
        #     inputs, outputs, shuffle=True, random_state=seed, test_size=test_size
        # )
        # train_info, test_info = train_test_split(filtered_data, shuffle=True, random_state=seed, test_size=test_size)
    elif design.startswith('repo-based') or design.startswith('source-test'):
        repos = pd.read_csv('./repos_3.csv', index_col=0)
        target_repos = repos[repos['category'] == 'target']

        input_repo = defaultdict(list)
        output_repo = defaultdict(list)
        filtered_instance_repo = defaultdict(list)
        for input_instance, output_instance, filtered_data_instance in zip(inputs, outputs, filtered_data):
            this_repo = filtered_data_instance.repo
            if select_repo and this_repo != select_repo:
                continue
            input_repo[this_repo].append(input_instance)
            output_repo[this_repo].append(output_instance)
            filtered_instance_repo[this_repo].append(filtered_data_instance)

        train_inputs, train_labels, train_info = list(), list(), list()
        test_inputs, test_labels, test_info = list(), list(), list()
        for repo in input_repo:
            if not (target_repos['repo'] == repo).any():  # (source)
                train_inputs += input_repo[repo]
                train_labels += output_repo[repo]
                train_info += filtered_instance_repo[repo]
            else:  # target
                if len(input_repo[repo]) < 2:
                    train_inputs += input_repo[repo]
                    train_labels += output_repo[repo]
                    train_info += filtered_instance_repo[repo]
                    continue
                if not no_split:
                    this_train_input, this_test_input, this_train_output, this_test_output, this_train_fi, this_test_fi = \
                        train_test_split(input_repo[repo], output_repo[repo], filtered_instance_repo[repo],
                                         shuffle=True, random_state=seed, test_size=0.20)
                    test_inputs += this_test_input
                    test_labels += this_test_output
                    test_info += this_test_fi
                    if design.endswith('included'):
                        train_inputs += this_train_input
                        train_labels += this_train_output
                        train_info += this_train_fi
                else:
                    train_inputs += input_repo[repo]
                    train_labels += output_repo[repo]
                    train_info += filtered_instance_repo[repo]
    else:
        print(f'wrong design argument {design}')
        return
    if not no_valid and len(train_inputs) > 1:
        # print(
        #     f'train size: {len(train_inputs)} | test size: {len(test_inputs)} | ratio: {len(test_inputs) / (len(test_inputs) + len(train_inputs)):.2f}')
        val_size = 0.25  # if len(train_inputs) >= 10 else 1 / len(train_inputs)
        train_inputs, val_inputs, train_labels, val_labels = train_test_split(
            train_inputs, train_labels, shuffle=True, random_state=seed, test_size=val_size
        )
        train_info, val_info = train_test_split(train_info, shuffle=True, random_state=seed, test_size=val_size)
    else:
        val_inputs, val_labels, val_info = [], [], []

    if design.startswith('source-test'):
        _, test_inputs, _, test_labels, _, test_info = train_test_split(val_inputs, val_labels, val_info,
                                                               random_state=seed, test_size=0.15)

    return (
        train_inputs,
        train_labels,
        val_inputs,
        val_labels,
        test_inputs,
        test_labels,
        train_info,
        val_info,
        test_info,
    )


def create_data(data: List[DataPoint], linter_warnings: List[str], include_warning: bool, design: str,
                select_repo=None, back_translation=False, no_split=False):
    train: List[str] = []
    train_labels: List[str] = []
    val: List[str] = []
    val_labels: List[str] = []

    test: DefaultDict[str, List[str]] = defaultdict(list)
    test_labels: DefaultDict[str, List[str]] = defaultdict(list)
    n_test_samples = 0

    train_info: List[DataPoint] = []
    val_info: List[DataPoint] = []
    test_info: DefaultDict[str, List[DataPoint]] = defaultdict(list)
    print(f'splitting by : {design}')
    for warning in linter_warnings:
        filtered_data = filter_rule(data, warning)
        (train_w, train_w_labels, val_w, val_w_labels, test_w, test_w_labels, train_w_info, val_w_info, test_w_info,) \
            = split_filtered(filtered_data, include_warning, design, select_repo=select_repo,
                             back_translation=back_translation, no_split=no_split)

        train += train_w
        train_labels += train_w_labels

        val += val_w
        val_labels += val_w_labels

        train_info += train_w_info
        val_info += val_w_info

        test[warning] = test_w
        test_labels[warning] = test_w_labels

        test_info[warning] = test_w_info

        n_test_samples += len(test_w)
    print("train size: {}\nval size: {}\ntest size: {}"
          .format(len(train), len(val), n_test_samples))
    return train, train_labels, val, val_labels, test, test_labels, train_info, val_info, test_info


def create_data_tbug(data: List[DataPoint], linter_warnings: List[str], include_warning: bool, design: str,
                     select_repo=None, back_translation=False, no_split=False):
    train: DefaultDict[str, List[str]] = defaultdict(list)
    train_labels: DefaultDict[str, List[str]] = defaultdict(list)

    test: DefaultDict[str, List[str]] = defaultdict(list)
    test_labels: DefaultDict[str, List[str]] = defaultdict(list)
    n_test_samples = 0

    train_info: DefaultDict[str, List[DataPoint]] = defaultdict(list)
    test_info: DefaultDict[str, List[DataPoint]] = defaultdict(list)
    print(f'splitting by : {design}')
    for warning in linter_warnings:
        filtered_data = filter_rule(data, warning)
        (train_w, train_w_labels, _, _, test_w, test_w_labels, train_w_info, _, test_w_info,) \
            = split_filtered(filtered_data, include_warning, design, select_repo=select_repo,
                             back_translation=back_translation, no_split=no_split, no_valid=True)

        train[warning] += train_w
        train_labels[warning] += train_w_labels
        train_info[warning] += train_w_info

        test[warning] = test_w
        test_labels[warning] = test_w_labels
        test_info[warning] = test_w_info

        n_test_samples += len(test_w)
    print("train size: {}\ntest size: {}"
          .format(len(train), n_test_samples))
    return train, train_labels, test, test_labels, train_info, test_info


class BugFixDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: BatchEncoding, targets: BatchEncoding):
        self.encodings = encodings
        self.target_encodings = targets

    def __getitem__(self, index: int) -> (int, Dict[str, Any]):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.target_encodings["input_ids"][index], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])


def create_dataset(
        inputs: List[str],
        labels: List[str],
        tokenizer: PreTrainedTokenizer,
        pad_truncate: bool,
        max_length=None,
) -> BugFixDataset:
    if max_length is not None:
        input_encodings = tokenizer(
            inputs, truncation=pad_truncate, padding=pad_truncate, max_length=max_length
        )
        label_encodings = tokenizer(
            labels, truncation=pad_truncate, padding=pad_truncate, max_length=max_length
        )
    else:
        input_encodings = tokenizer(
            inputs, truncation=pad_truncate, padding=pad_truncate, max_length=256
        )
        label_encodings = tokenizer(
            labels, truncation=pad_truncate, padding=pad_truncate, max_length=256
        )

    dataset = BugFixDataset(input_encodings, label_encodings)
    return dataset
