import os
import pickle as pk
from argparse import ArgumentParser

def gen_todo_list(directory, check = None):
    files = os.listdir(directory)
    todo_list = []
    for f in files:
        fullpath = os.path.join(directory, f)
        if os.path.isfile(fullpath):
            if check is not None:
                if check(f):
                    todo_list.append(fullpath)
            else:
                todo_list.append(fullpath)
    return todo_list


def dump(data, filename):
    with open(filename, 'wb') as f:
        pk.dump(data, f)

def load(filename):
    with open(filename, 'rb') as f:
        data = pk.load(f)
    return data

def get_args():
    parser = ArgumentParser()
    #parser.add_argument("pos1", help="positional argument 1")
    parser.add_argument("-m", "--mode", help="trainer mode [train/test]", dest="mode", default="train")
    parser.add_argument("-n", "--name", help="model name [model]", dest="model_name", default="model")
    parser.add_argument("-t", "--model_type", help="model type [cnn/sae]", dest="model_type", default="cnn")
    parser.add_argument("-tt", "--task_type", help="task type [app/class]", dest="task_type", default="app")
    parser.add_argument("-bs", "--batch-size", help="batch size [256]", dest="batch_size", default=256, type=int)
    parser.add_argument("-db", "--debug", help="debug [False]", dest="debug", action="store_true")
    parser.add_argument("-l", "--load", help="load [False]", dest="load", action="store_true")
    args = parser.parse_args()
    return args

from keras import backend as K

def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

