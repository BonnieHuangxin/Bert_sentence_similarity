from __future__ import absolute_import, division, print_function
import argparse
import csv
import logging
import time
import os
import random
import sys
import re
import json
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import torch.nn.functional as F

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def _read_txt(cls, line, quotechar=None):
        """Reads a tab separated value file."""
        lines = []
        line = line.strip()
        label = line[-1]
        text_1 = line[:-1].strip().split('#')[0]
        text_2 = line[:-1].strip().split('#')[1]
        ll_line = [text_1,text_2,label]
        lines.append(ll_line)
        return lines


class SimProcessor(DataProcessor):

    def get_dev_examples(self, line):

        return self._create_examples(self._read_txt(line), "dev")
        #序号、sen1、sen2、类别


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


    #返回所有的类别
    def get_labels(self):
        """See base class."""
        return ["0", "1"]


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


class Predict:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        c_bert_model = "./tmp_chinese/mrpc_output/"
        raw_bert_model = "./models/chinese_L-12_H-768_A-12"
        num_labels = 2
        self.tokenizer = BertTokenizer.from_pretrained(raw_bert_model)
        self.model = BertForSequenceClassification.from_pretrained(c_bert_model, num_labels=num_labels)
        self.model.to(device)

    def predict(self, line):
        processors = {
            # "cola": ColaProcessor,
            # "mnli": MnliProcessor,
            "mrpc": SimProcessor
        }

        num_labels_task = {
            "mrpc": 2,
        }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        processor = processors['mrpc']()
        label_list = processor.get_labels()
        max_seq_length = 128
        eval_batch_size = 8

        # tokenizer = BertTokenizer.from_pretrained(raw_bert_model)
        # model = BertForSequenceClassification.from_pretrained(c_bert_model, num_labels=num_labels)
        # model.to(device)

        test_line = line + "\t1"


        eval_examples = processor.get_dev_examples(test_line)
        eval_features = convert_examples_to_features(eval_examples, label_list, max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

        self.model.eval()

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = self.model(input_ids, segment_ids, input_mask, label_ids)
                logits = self.model(input_ids, segment_ids, input_mask)
            score = F.softmax(logits, 1)
            maximum_probability = score.detach().cpu().numpy()[0].max()
            print(maximum_probability)
            logits = logits.detach().cpu().numpy()[0]
            res = np.argmax(logits)
            # return res

            id2Senti = {
                "0":'不同',
                "1":'相同',
            }

            result = {
                'content': line,
                'result': id2Senti[str(res)],
                'probability': str(round(100*maximum_probability,2))+'%'
            }
            # return result
            return json.dumps(result, ensure_ascii=False)
            # label_ids = label_ids.to('cpu').numpy()


if __name__ == "__main__":
    p = Predict()
    print(p.predict("你多大了？#你的年龄是多少？"))
    #input_file = './chinese_data/data_dev.txt'
    #sequence = read_txt(input_file)
    #print(time.strftime("%H:%M:%S"))
    #for i in range(len(sequence)):
        #print(p.predict(sequence[i]))
    #print(time.strftime("%H:%M:%S"))
