# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """


import logging
import os
import pickle
from data_util import corpus2list

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, spacy_w):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.spacy_w = spacy_w


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

class My_Input(object):
    def __init__(self, guid, words, labels,input_ids, input_mask, segment_ids, label_ids):
        self.guid = guid
        self.words = words
        self.labels = labels
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(guid="%s-%d".format(mode, guid_index), words=words, labels=labels))
    return examples

def my_read_examples_from_file(data_dir, mode):
    # if mode == "train":
    #     path = os.path.join(data_dir, "data_train_task1.pkl")
    # elif mode == "dev":
    #     path = os.path.join(data_dir, "data_dev_task1.pkl")
    # else:
    #     path = os.path.join(data_dir, "data_test_task1.pkl")
    path = os.path.join(data_dir, "data_{}_task1.pkl".format(mode))
    datasets = pickle.load(open(path, "rb"))
    ids = datasets[0]
    texts = datasets[1]
    labels = datasets[2]  # 0；O， propganda:1
    examples = []
    for sent_id, sent, label in zip(ids, texts, labels):
        if len(sent) ==0:
            continue
        new_label = []
        for lab in label:
            if lab == 0:
                new_label.append("O")
            else:
                new_label.append(lab)
        examples.append(InputExample(guid="{}-{}".format(mode, sent_id), words=sent, labels=new_label))
    # for sent_id, sent, label in zip(ids, texts, labels):
    #     new_words = []
    #     new_sent
    return examples


def read_preproces_data(mode, data_dir, labels):
    data_path = os.path.join(data_dir, "my_{}_task1.pkl".format(mode))
    data_dict = pickle.load(open(data_path, "rb"))
    # labels = get_labels("BIO")
    label2id = {label: idx for idx, label in enumerate(labels)}
    # sent_ids, sent_texts, sent_labels, sent_spacys = corpus2list(p2id=label2id,
    #                                                              ids=data_dict["ID"],
    #                                                              texts=data_dict["Text"],
    #                                                              labels=data_dict["Label"])
    if mode != "train":
        data_dict["Label"] = [[] for _ in range(len(data_dict["ID"]))]
    dataset = corpus2list(p2id=label2id,
                          bio=False,
                          ids=data_dict["ID"],
                          texts=data_dict["Text"],
                          labels=data_dict["Label"])
    return dataset

def read_and_convert_data(
        mode, datasets,
        label_list,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
):
    # path = os.path.join(data_dir, "data_{}_task1.pkl".format(mode))
    # datasets = pickle.load(open(path, "rb"))
    ids = datasets[0]
    texts = datasets[1]
    labels = datasets[2]  # 0；O， propganda:1
    spacys = datasets[3]

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    examples = []

    for data_idx, (sent_id, sent, label, spacy) in enumerate(zip(ids, texts, labels, spacys)):
        if data_idx % 5000 == 0:
            logger.info("Writing example %d of %d", data_idx, len(ids))
        # if data_idx == 1409:
        #     print("1")
        new_label = []
        new_sent = []
        new_spacy = []
        tokens = []
        label_ids = []
        for idx, (lab, word) in enumerate(zip(label, sent)):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) == 0:
                continue
            tokens.extend(word_tokens)
            # if lab == 0:
            #     lab = "O"
            new_label.append(lab)
            new_sent.append(sent[idx])
            new_spacy.append(spacy[idx])
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            # label_ids.extend([label_map[lab]] + [pad_token_label_id] * (len(word_tokens) - 1))
            label_ids.extend([lab] + [pad_token_label_id] * (len(word_tokens) - 1))
        if len(new_label) != 0:
            assert len(new_spacy) == len(new_sent)
            assert len(new_sent) == len(new_label)
            examples.append(InputExample(guid="{}-{}".format(mode, sent_id), words=new_sent,
                                         labels=new_label, spacy_w=new_spacy))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        if len(tokens) == 0:
            continue
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if data_idx < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", examples[data_idx].guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids)
        )

    return examples, features


def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            # if len(word_tokens) == 0:
            #     continue
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids)
        )
    return features


def get_labels(label):
    if label == 2:
        # with open(path, "r") as f:
        #     labels = f.read().splitlines()
        # if "O" not in labels:
        #     labels = ["O"] + labels
        labels = ["O", "B", "I", "E"]
        return labels
    elif label == 1:
        labels = ["O", "B", "I"]
        return labels
    else:
        return ["O", "B"]


def get_propgranda_span(prediction: list, examples: list):
    assert len(prediction) == len(examples)

    spans = []
    for i, pred in enumerate(prediction):
        sent_id = examples[i].guid.split("-")[-1]
        sent_wd = examples[i].words
        sent_sp = examples[i].spacy_w

        try:
            assert len(pred) == len(sent_wd)
        except AssertionError:
            print(i)
            continue
        prev = "O"
        span = []
        span_len = 0
        for j, pred_label in enumerate(pred):
            if pred_label != "O" and prev == "O":
                span_start = sent_sp[j].idx
                span_len += 1
                prev = pred_label
            if pred_label =="O" and prev != "O":
                span_end = sent_sp[j-1].idx + len(sent_sp[j-1])
                span.append((sent_id, span_start, span_end))
                span_len = 0
                prev = pred_label
            if j >= len(pred) - 1 and pred_label != "O":
                if span_len == 1:
                    span_end = sent_sp[j].idx + len(sent_sp[j])
                    span.append((sent_id, span_start, span_end))
                # else:
                #     span_start =
        spans.extend(span)
    return spans

def data_stitic(examples, mode):
    max_length = 0
    min_length = 200
    for example in examples:
        exam_len = len(example.words)
        if exam_len > max_length:
            max_length = exam_len
        if exam_len < min_length:
            min_length = exam_len
    print("{} data:".format(mode))
    print("the number of sentences: {}".format(len(examples)))
    print("the max length: {}".format(max_length))
    print("the min length: {}".format(min_length))

# if __name__ == '__main__':
#     # a = 2
#     # b = 4
#     # c = 5
#     # print([a] + [b] * c)
#     test_class = []
#     for i in range(10):
#         obj = InputExample(guid=str(i),
#                            words=["Age", "at", "three"],
#                            labels=["O", "O", "pra"],
#                            spacy_w=["Age", "at", "three"])
#         test_class.append(obj)
#     with open("test-obj.pkl","wb") as f:
#         # f.write(test_class)
#         pickle.dump(test_class,f)
#     print("writed")
#     with open("test-obj.pkl", 'rb') as fl:
#         data = pickle.load(fl)
#     for i in data:
#         print(i.guid)
if __name__ == '__main__':
    # path = "../datasets-v2/datasets/train-labels-task1-span-identification"
    path = "../mid_out1"
    data = read_preproces_data(mode="train", data_dir=path)
    print("!")