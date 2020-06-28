import pickle
import spacy
import pandas as pd
import pathlib
from pathlib import Path
from spacy.tokens import Doc, Token
import torch
nlp = spacy.load('en')

# def parse_label(label_path:str):
#     labels = []
#     f = Path(label_path)
#     if not f.exists():
#         return labels
#     for line in open(label_path):
#         parts = line.strip().split('\t')
#         labels.append((parts[1], parts[2]))
#     return sorted(labels)

def parse_label(label_path: str, binary: str = None) -> list:
    # idx, type, start, end
    labels = []
    f = Path(label_path)
    if not f.exists():
        return labels
    for line in open(label_path, encoding='utf-8-sig'):
        parts = line.strip().split('\t')
        if binary:
            if binary == 'Propaganda':
                labels.append((int(parts[1]), int(parts[2]), binary))
            # else:
            #     labels.append((int(parts[1]), int(parts[2]), 'Nopropaganda'))
        else:
            labels.append((int(parts[2]), int(parts[3]), parts[1]))

    return sorted(labels)

def read_task1(path:str, isLabels:bool, binary:str) ->list:
    directory = pathlib.Path(path)
    ids = []
    texts = []
    labels = []
    for f in directory.glob('*.txt'):
        id = f.name.replace('article', '').replace('.txt', '')
        ids.append(id)
        texts.append(f.read_text(encoding='utf-8-sig'))
        if isLabels:
            labels.append(parse_label(
                f.as_posix().replace('.txt', '.task1-SI.labels'), binary))

    docs = list(nlp.pipe(texts))

    return [ids, docs, labels]

def get_labels():
    pass


def safe_list_get(l, idx, default=0):
    try:
        return l[idx]
    except IndexError:
        return [0, 0, 0]

def bert_list(p2id: dict, doc: Doc,
              doc_labels: list, ids: list, binary: bool=True, bio: bool = True) -> tuple:
    token_idx = 0
    labels_idx = 0
    tokensh = []
    labelsh = []
    tlabel = []
    tspacyt = []
    ttoken = []
    bertids = []
    flagger = 0
    spacytokens = []
    while token_idx < len(doc):
        current_token: Token = doc[token_idx]
        start_token_idx = token_idx
        current_label = safe_list_get(doc_labels, labels_idx)
        # advance token until it is within the label
        if (str(current_token)[:1] == '\n'):
            flagger = 0
            if ttoken:
                spacytokens.append(tspacyt)
                tokensh.append(ttoken)
                labelsh.append(tlabel)
                bertids.append(ids)
            tlabel = []
            tspacyt = []
            ttoken = []
            token_idx += 1
            continue
        if current_token.idx < current_label[0] or current_label[2] == 0:
            # Uncomment to get backtrack
            # if flagger == 0:
            ttoken.append(str(current_token))
            tspacyt.append(current_token)
            tlabel.append(p2id["O"])
            flagger = flagger - 1
            if flagger < 0:
                flagger = 0
            token_idx += 1
            continue

        flagger = 0
        first = True
        while current_token.idx < current_label[1]:
            if (str(current_token)[:1] == '\n'):
                if ttoken:
                    spacytokens.append(tspacyt)
                    tokensh.append(ttoken)
                    labelsh.append(tlabel)
                    bertids.append(ids)
                tlabel = []
                tspacyt = []
                ttoken = []

            else:
                ttoken.append(str(current_token))
                tspacyt.append(current_token)

                if first:
                    # tlabel.append(p2id[current_label[2]])
                    tlabel.append(p2id["B"])
                else:
                    tlabel.append(p2id["I"])

            token_idx += 1
            if token_idx >= len(doc):
                break
            current_token = doc[token_idx]
            flagger = flagger + 1
            if bio:
                first = False
            else:
                first = True
        # advance label
        labels_idx += 1

        # revert token_idx because the labels might be intersecting. Uncomment to get backtrack.
        # token_idx = start_token_idx

    def BIO2BIOE(bio_labels, p2id):
        for bio_label in bio_labels:
            for idx, label in enumerate(bio_label):
                if label != p2id["I"]:
                    continue
                try:
                    if bio_label[idx + 1] == p2id["O"]:
                        bio_label[idx] = p2id["E"]
                except:
                    bio_label[idx] = p2id["E"]

        return bio_labels
    # labelsh = BIO2BIOE(labelsh, p2id)
    return bertids, tokensh, labelsh, spacytokens

def load_technique_names_from_file(filename: str) -> list:
    with open(filename, "r") as f:
        return [line.rstrip() for line in f.readlines()]


def settings(tech_path: str, label: str = None, bio: bool = False) -> tuple:
    prop_tech_bio = None
    if label:
        prop_tech = [label]
    else:
        prop_tech = load_technique_names_from_file(tech_path)

    if bio:
        prop_tech_inside = prop_tech
        prop_tech_begin = ["B-" + tech for tech in prop_tech]
        prop_tech_inside = ["I-" + tech for tech in prop_tech_inside]
        prop_tech_bio = prop_tech_begin + prop_tech_inside
    # TODO prop_tech or prop_tech_e
    offset = len(prop_tech)
    hash_token = offset + 1
    end_token = offset + 2
    # Insert "outside" element
    prop_tech.insert(0, "O")
    if prop_tech_bio:
        prop_tech_bio.insert(0, "O")
    p2id = {y: x for (x, y) in enumerate(prop_tech)}
    return prop_tech_bio, prop_tech, hash_token, end_token, p2id


def pad_sequences(sequences: list, batch_first: bool = True,
                  padding_value: int = 0, max_len: int = 0):
    tmp = torch.Tensor(sequences[0])
    max_size = tmp.size()
    trailing_dims = max_size[1:]

    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = tmp.data.new(*out_dims).fill_(padding_value)
    for i, list in enumerate(sequences):
        tensor = torch.Tensor(list)
        length = tensor.size(0)
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor.long().numpy()


def reg_encoding(cleaned: list, labels: list, hash_token, end_token) -> list:
    label_l = []
    for oindex, x in enumerate(cleaned):
        tlist = []
        for index, j in enumerate(x):
            for s in j:
                if s[0] == '#':
                    tlist.append(hash_token)
                else:
                    tlist.append(labels[oindex][index])
        label_l.append(tlist)
    return label_l


def bio_encoding(cleaned: list, labels: list, hash_token) -> list:
    offset = 1

    label_l = []
    for oindex, x in enumerate(cleaned):
        tlist = []
        prev = labels[oindex][0]
        for index, j in enumerate(x):
            # if index==30:
            # ipdb.set_trace()
            for s in j:
                if s[0] == '#':
                    tlist.append(hash_token)
                else:
                    if (index == 0 and labels[oindex][index] != 0):
                        tlist.append(labels[oindex][index] + offset)
                        prev = labels[oindex][index]
                    if (prev != labels[oindex][index] and labels[oindex][index] != 0):
                        tlist.append(labels[oindex][index] + offset)
                        prev = labels[oindex][index]
                    else:
                        tlist.append(labels[oindex][index])
                        prev = labels[oindex][index]
        label_l.append(tlist)
    return label_l

def corpus2list(p2id: dict, ids: list, texts: list,
                labels: list, binary_calss: bool = False, bio: bool = False) -> tuple:
    print(p2id)
    berti, bertt, bertl, berts = zip(*[bert_list(p2id, d, l, idx, binary_calss, bio)
                                       for d, l, idx in zip(texts, labels, ids)])
    flat_list_text = [item for sublist in bertt for item in sublist]
    flat_list_label = [item for sublist in bertl for item in sublist]
    flat_list_id = [item for sublist in berti for item in sublist]
    flat_list_spacy = [item for sublist in berts for item in sublist]
    # print (flat_list_text[0])
    return flat_list_id, flat_list_text, flat_list_label, flat_list_spacy

def concatenate_list_data(cleaned: list) -> list:
    result= []
    for element in cleaned:
        result += element
    return result

def make_set(p2id, data_dir: str, tokenizer, single_class: bool,
             hash_token, end_token, bio: bool = False, maxLen:int = 192) -> tuple:

    data_dict = pickle.load(open(data_dir, "rb"))

    dataset = corpus2list(p2id, data_dict["ID"], data_dict["Text"],
                          data_dict["Label"], single_class, bio)
    my_data = [dataset[0], dataset[1], dataset[2]]
    with open('./../mid_out/data_eval_task1.pkl', 'wb')as handle:
        pickle.dump(my_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Shuffle samples
    # dataset = dataset.sample(frac=1)
    terms = list(dataset[1])
    labels = list(dataset[2])

    cleaned = [[tokenizer.tokenize(words) for words in sent] for sent in terms]
    tokenized_texts = [concatenate_list_data(sent) for sent in cleaned]
    if bio:
        label_l = bio_encoding(cleaned, labels)
    else:
        label_l = reg_encoding(cleaned, labels, hash_token, end_token)

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              padding_value=0.0, max_len=maxLen)

    tags = pad_sequences(label_l, padding_value=end_token, max_len=maxLen)
    attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]

    return input_ids, tags, attention_masks, label_l


def make_val_set(p2id, data_dir: str, tokenizer, single_class: str,
                 hash_token, end_token, bio: bool = False, maxLen:int=192) -> tuple:
    # dataset = pd.read_csv(data_dir, sep='\t', header=None, converters={1:ast.literal_eval, 2:ast.literal_eval})
    data_dict = pickle.load(open(data_dir, "rb"))
    if not bio:
        dataset = corpus2list(p2id, data_dict["ID"], data_dict["Text"],
                              data_dict["Label"], single_class, bio)
    my_data = [dataset[0], dataset[1], dataset[2]]
    with open('./../mid_out/data_eval_task1.pkl', 'wb')as handle:
        pickle.dump(my_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Shuffle samples
    # dataset = dataset.sample(frac=1)
    ids = (dataset[0])
    terms = (dataset[1])
    labels = (dataset[2])
    spacy = (dataset[3])
    cleaned = [[tokenizer.tokenize(words) for words in sent] for sent in terms]
    tokenized_texts = [concatenate_list_data(sent) for sent in cleaned]

    if bio:
        label_l = bio_encoding(cleaned, labels)
    else:
        label_l = reg_encoding(cleaned, labels, hash_token, end_token)

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              padding_value=0.0, max_len=maxLen)

    tags = pad_sequences(label_l, padding_value=end_token, max_len=maxLen)
    attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]

    return input_ids, tags, attention_masks, cleaned, ids, terms, spacy, label_l

def get_char_level(flat_list_i: list, flat_list_s: list,
                   predictions_sample: list, cleaned: list,
                   hash_token: int, end_token: int, prop_tech) -> pd.DataFrame:
    # counter = 0
    # for x in predictions_sample:
    #     for j in x:
    #         if j == 1:
    #             counter = counter + 1
    #             break
    # print(counter)
    pred = []
    for oindex, x in enumerate(cleaned):
        index = 0
        tlist = []
        for iindex, j in enumerate(x):
            # print (j)
            # print(index)
            tlist.append(predictions_sample[oindex][index])
            length = len(j)
            index = index + length
            # print ("Token: ", j, "-----  Assigned: ", predictions_sample[oindex][index])
        pred.append(tlist)

    # tpred = pred
    # pred = []
    # for x in tpred:
    #     tlist = []
    #     for j in x:
    #         if j in [hash_token, end_token]:
    #             continue
    #         tlist.append(j)
    #     pred.append(tlist)
    # counter = 0
    # for x in predictions_sample:
    #     for j in x:
    #         if j == 1:
    #             counter = counter + 1
    #             break
    # print("Counter check: ", counter)
    lists = []
    liste = []
    listp = []
    listid = []

    for i, x in enumerate(pred):
        a = flat_list_s[i]
        b = flat_list_i[i]
        id_text, spans = get_spans(a, x, i, b, hash_token, end_token, prop_tech)
        if spans:
            for span in spans:
                listid.append(id_text)
                liste.append(span[2])
                lists.append(span[1])
                listp.append(span[0])
    df = {"ID": listid, "P": listp, "s": lists, "liste": liste}
    df = pd.DataFrame(df)
    return df


def get_spans(a: list, labelx: list, i: int, id_text: str, hash_token, end_token, prop_tech):
    # if i==35:
    # ipdb.set_trace()
    spans = []
    span_len = 0
    prev = 0
    for i, x in enumerate(labelx):
        # End if last index\

        if x == end_token:
            continue
        if i >= len(a) - 1:
            if x != 0:
                # if prev element isn't equal to current and not O
                if prev != x and prev != 0:
                    span_e = a[i - 1].idx + len(a[i - 1])
                    span_len = 0
                    spans.append([prop_tech[labelx[i - 1]], span_f, span_e])
                    prev = x
                    span_f = a[i].idx
                    span_len = span_len + 1
                if span_len == 0:
                    span_f = a[i].idx
                    span_len = span_len + 1
                    prev = x
                    if (i >= len(labelx) - 1):
                        span_e = a[i].idx + len(a[i])
                        span_len = 0
                        spans.append([prop_tech[labelx[i]], span_f, span_e])
                        continue
                else:
                    span_e = a[i].idx + len(a[i])
                    span_len = 0
                    spans.append([prop_tech[labelx[i]], span_f, span_e])
                    continue

            else:
                prev = x
                if (span_len != 0):
                    span_e = a[i - 1].idx + len(a[i - 1])
                    span_len = 0
                    spans.append([prop_tech[labelx[i - 1]], span_f, span_e])
                    continue
        if x == hash_token:
            continue
        if x != 0:
            # Check if prev element was same as current or equal to O
            if prev != x and prev != 0:
                span_e = a[i - 1].idx + len(a[i - 1])
                span_len = 0
                spans.append([prop_tech[labelx[i - 1]], span_f, span_e])
                prev = x
                span_f = a[i].idx
                span_len = span_len + 1
            if span_len == 0:
                span_f = a[i].idx
                span_len = span_len + 1
                prev = x
                if (i >= len(labelx) - 1):
                    span_e = a[i].idx + len(a[i])
                    span_len = 0
                    spans.append([prop_tech[labelx[i]], span_f, span_e])
                    continue
            else:
                if (i >= len(labelx) - 1):
                    span_e = a[i].idx + len(a[i])
                    span_len = 0
                    spans.append([prop_tech[labelx[i]], span_f, span_e])
                    continue
                span_len = span_len + 1

        else:
            prev = x
            if (span_len != 0):
                span_e = a[i - 1].idx + len(a[i - 1])
                span_len = 0
                spans.append([prop_tech[labelx[i - 1]], span_f, span_e])
                continue
            if (i >= len(labelx) - 1):
                # span_e= a[i].idx + len(a[i])
                # span_len = 0
                # spans.append([span_f, span_e])
                continue
    if spans:
        return id_text, spans
    else:
        return (0, [])


if __name__ == '__main__':
    path = "../datasets-v2/datasets/train-labels-task1-span-identification"
