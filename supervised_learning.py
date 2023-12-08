from statistics import mean, median
import numpy as np
import copy
import gc
import os

from Prediction import ML

import warnings
warnings.filterwarnings("ignore")

def ensure_dir(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def process_SS_file(path_file_SS):
    with open(path_file_SS, 'r') as file_SS:
        dict_SS = {}
        for line in file_SS:
            line = line[:-1]
            split1 = line.split('\t')
            ent1 = split1[0].split('/')[-1]
            ent2 = split1[1].split('/')[-1]
            SS = split1[2:]
            SS_floats = [float(i) for i in SS]
            dict_SS[(ent1, ent2)] = SS_floats
    return dict_SS


def process_dataset(path_dataset_file):
    with open(path_dataset_file, 'r') as dataset:
        list_labels, list_pairs = [], []
        for line in dataset:
            split1 = line.split('\t')
            ent1, ent2 = split1[0], split1[1]
            label = int(split1[2][:-1])
            list_labels.append(label)
            list_pairs.append([ent1, ent2])
    return list_labels, list_pairs


def process_indexes_partition(file_partition):
    with open(file_partition, 'r') as file_partitions:
        indexes_partition = []
        for line in file_partitions:
            indexes_partition.append(int(line[:-1]))
    return indexes_partition


def main(path_pairs_file, path_partition, path_output_ss, aspects, path_output, n_partition):
    algorithms = ['DT6', 'DT', 'GP', 'GP6x', 'RF', 'XGB']
    list_labels, list_pairs = process_dataset(path_pairs_file)
    dict_SS = process_SS_file(path_output_ss)
    list_ss = [dict_SS[(ent1, ent2)] for (ent1,ent2) in list_pairs]
    n_pairs = len(list_labels)

    for run in range(1, n_partition + 1):
        file_partition = path_partition + str(run) + '.txt'
        test_index = process_indexes_partition(file_partition)
        train_index = list(set(range(0, n_pairs)) - set(test_index))
        list_labels, list_ss, list_pairs = np.array(list_labels), np.array(list_ss), np.array(list_pairs)
        y_train, y_test, pairs_test = list(list_labels[train_index]), list(list_labels[test_index]), list(list_pairs[test_index])
        X_train, X_test = list(list_ss[train_index]), list(list_ss[test_index])

        for alg in algorithms:
            ensure_dir(path_output  + "/" + alg + "/")
            path_output_predictions = path_output + "/" + alg + "/Predictions__" + "Run" + str(run) + ".txt"
            path_output_evaluation_predictions = path_output + "/" + alg + "/EvaluationPredictions__" + "Run" + str(run) + ".txt"
            path_output_model = path_output + "/" + alg + "/" + "/Model__" + "Run" + str(run) + ".txt"
            path_output_explanations =  path_output + "/" + alg + "/" + "/Explanations__" + "Run" + str(run) + ".txt"

            if alg == "DT":
                ML.performance_DT(dict_SS, pairs_test, X_train, X_test, y_train, y_test, aspects, path_output_predictions, path_output_model,
                 path_output_evaluation_predictions, path_output_explanations)

            elif alg == "DT6":
                ML.performance_DT(dict_SS, pairs_test, X_train, X_test, y_train, y_test, aspects, path_output_predictions, path_output_model,
                 path_output_evaluation_predictions, path_output_explanations, d=6)

            elif alg == "GP":
                ML.performance_GP(pairs_test, X_train, X_test, y_train, y_test, aspects, path_output_predictions, path_output_model,
                 path_output_evaluation_predictions, path_output_explanations)

            elif alg == "GP6x":
                ML.performance_GP6x(pairs_test, X_train, X_test, y_train, y_test, aspects, path_output_predictions, path_output_model,
                 path_output_evaluation_predictions, path_output_explanations)

            elif alg == "RF":
                ML.performance_RF(pairs_test, list_ss, list_labels, X_train, X_test, y_train, y_test, aspects, path_output_predictions, path_output_model,
                 path_output_evaluation_predictions, path_output_explanations)

            elif alg == "XGB":
                ML.performance_XGB(pairs_test, list_ss, list_labels, X_train, X_test, y_train, y_test, aspects, path_output_predictions, path_output_model,
                 path_output_evaluation_predictions, path_output_explanations)

