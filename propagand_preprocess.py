import os
import pickle
from data_util import *
from sklearn.model_selection import train_test_split


def train_dev_spilte(path: str, spilte=0.15):
    data_dict = pickle.load(open(path, "rb"))
    id = data_dict["ID"]
    texts = data_dict["Text"]
    lables = data_dict["Label"]

    train_id, dev_id, train_text, dev_text, train_lable, dev_lable = \
        train_test_split(id, texts, lables, test_size=spilte, shuffle=True)
    train_dict = {"ID": train_id, "Text": train_text, "Label": train_lable}
    dev_dict = {"ID": dev_id, "Text": dev_text, "Label": dev_lable}
    with open('./../mid_out/my_train.pkl', 'wb') as handle:
        pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./../mid_out/my_dev.pkl', 'wb')as handle:
        pickle.dump(dev_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main(task1_path: str, islabel: bool, binary: str):
    ids, texts, labels = read_task1(task1_path, islabel, binary)
    # ds = {"ID": ids, "Text": texts, "Label": labels}
    train_id, dev_id, train_text, dev_text, train_lable, dev_lable = \
        train_test_split(ids, texts, labels, test_size=0.15, shuffle=True)
    train_dict = {"ID": train_id, "Text": train_text, "Label": train_lable}
    dev_dict = {"ID": dev_id, "Text": dev_text, "Label": dev_lable}
    with open('./../mid_out/my_train_task1.pkl', 'wb') as handle:
        pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./../mid_out/my_dev_task1.pkl', 'wb')as handle:
        pickle.dump(dev_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def eav_test1(text_path: str, islabel=False, binary: str = None):
    ids, texts, labels = read_task1(text_path, islabel, binary)
    test_dict = {"ID": ids, "Text": texts, "Label": labels}
    with open('./../mid_out/test_task1.pkl', 'wb')as handle:
        pickle.dump(test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def new_process(path: str = "./../mid_out/", mode: str = 'train'):
    # if mode == "train":
    pkl_path = os.path.join(path, "my_{}_task1.pkl".format(mode))
    data_dict = pickle.load(open(pkl_path, "rb"))
    return data_dict


if __name__ == '__main__':
    # train_path = './../mid_out/train-train.pkl'
    # train_dev_spilte(train_path)
    # binary = "Propaganda"
    # task1_path = "./../datasets-v2/datasets/train-labels-task1-span-identification"
    # main(task1_path, True, binary)
    evaluate_path = "./../datasets-v2/datasets/dev-articles"
    eav_test1(text_path=evaluate_path)
    # techniques = "./../tools-v2/tools/data/propaganda-techniques-binary.txt"
    # binaryLabel = "Propaganda"
    # bio = False
    # mode = "test"
    # binaryClass = True
    # prop_tech_e, prop_tech, hash_token, end_token, p2id = \
    #     settings(techniques, binaryLabel, bio)
    # data_dict = new_process(mode=mode)
    # if mode == "test":
    #     data_dict["Label"] = [[] for _ in range(len(data_dict["ID"]))]
    # dataset = corpus2list(p2id, data_dict["ID"], data_dict["Text"],
    #                       data_dict["Label"], binaryClass, bio)
    # my_data = [dataset[0], dataset[1], dataset[2]]
    # with open('./../mid_out/data_{}_task1.pkl'.format(mode), 'wb')as handle:
    #     pickle.dump(my_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
