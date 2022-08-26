from statistics import mean, median
import numpy as np
import copy
import gc
import os

from Prediction import ML

import warnings
warnings.filterwarnings("ignore")



def ensure_dir(path):
    """
    Check whether the specified path is an existing directory or not. And if is not an existing directory, it creates a new directory.
    :param path: path-like object representing a file system path;
    """
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)



def process_SS_file(path_file_SS):
    """
    Process the similarity file and returns a dictionary with the similarity values for each pair of ebtities.
    :param path_file_SS: similarity file path. The format of each line of the similarity file is "Ent1  Ent2    Sim_SA1 Sim_SA2 Sim_SA3 ... Sim_SAn";
    :return: dict_SS is a dictionary where the keys are tuples of 2 entities and the values are the similarity values taking in consideration different semantic aspects.
    """
    file_SS = open(path_file_SS, 'r')
    dict_SS = {}

    for line in file_SS:
        line = line[:-1]
        split1 = line.split('\t')

        ent1 = split1[0].split('/')[-1]
        ent2 = split1[1].split('/')[-1]
        SS = split1[2:]
        SS_floats = [float(i) for i in SS]
        dict_SS[(ent1, ent2)] = SS_floats

    file_SS.close()
    return dict_SS



def process_dataset(path_dataset_file):
    """
    Process the dataset file and returns a list with the proxy value for each pair of entities.
    :param path_dataset_file: dataset file path. The format of each line of the dataset file is "Ent1  Ent2    Proxy";
    :return: list_proxies is a list where each element represents a list composed by [(ent1, ent2), proxy].
    """
    dataset = open(path_dataset_file, 'r')
    list_labels = []

    for line in dataset:
        split1 = line.split('\t')
        ent1, ent2 = split1[0], split1[1]
        label = int(split1[2][:-1])
        list_labels.append([(ent1, ent2), label])

    dataset.close()
    return list_labels



def read_SS_dataset_file(path_file_SS, path_dataset_file):
    """
    Process the dataset file and the similarity file.
    :param path_file_SS: similarity file path. The format of each line of the similarity file is "Ent1  Ent2    Sim_SA1 Sim_SA2 Sim_SA3 ... Sim_SAn";
    :param path_dataset_file: dataset file path. The format of each line of the dataset file is "Ent1  Ent2    Proxy";
    :return: returns 4 lists.
    list_ents is a list of entity pairs in the dataset (each element of the list is a list [ent1, ent2]).
    list_SS is also a list of lists with the similarity values for each pair (each element of the list is a list [Sim_SA1,Sim_SA2,Sim_SA3,...,Sim_SAn]).
    list_SS_max_avg is a list of lists with the similarity values for each pair, including the average and the maximum (each element of the list is a list [Sim_SA1,Sim_SA2,Sim_SA3,...,Sim_SAn, Sim_AVG, Sim_MAX]).
    proxies is a list of proxy values for each pair in the dataset.
    """
    list_SS, list_SS_max_avg = [], []
    labels, list_ents = [], []

    dict_SS = process_SS_file(path_file_SS)
    list_labels = process_dataset(path_dataset_file)

    for (ent1, ent2), proxy in list_labels:

        SS_floats = dict_SS[(ent1, ent2)]
        max_SS = max(SS_floats)
        avg_SS = mean(SS_floats)

        list_ents.append([ent1, ent2])
        list_SS.append(SS_floats)
        labels.append(proxy)

        new_SS_floats = copy.deepcopy(SS_floats)
        new_SS_floats.append(avg_SS)
        new_SS_floats.append(max_SS)
        list_SS_max_avg.append(new_SS_floats)

    return list_ents, list_SS , list_SS_max_avg, labels



def process_indexes_partition(file_partition):
    """
    Process the partition file and returns a list of indexes.
    :param file_partition: partition file path (each line is a index);
    :return: list of indexes.
    """
    file_partitions = open(file_partition, 'r')
    indexes_partition = []
    for line in file_partitions:
        indexes_partition.append(int(line[:-1]))
    file_partitions.close()
    return indexes_partition



def run_cross_validation(algorithms, path_file_SS, path_dataset_file, dataset_name, path_results, n_partition, path_partition, SSM, aspects):
    """
    Run machine learning algorithms to learn the best combination of semantic aspects.
    :param algorithms: list of the algorithm (options:"GP", "LR", "XGB", "RF", "DT", "KNN", "BR", "MLP");
    :param proxy: proxy name (e.g. SEQ, PFAM, PhenSeries, PPI);
    :param path_file_SS: similarity file path. The format of each line of the similarity file is "Ent1  Ent2    Sim_SA1 Sim_SA2 Sim_SA3 ... Sim_SAn";
    :param path_dataset_file: dataset file path. The format of each line of the dataset file is "Ent1  Ent2    Proxy";
    :param dataset_name: name of the dataset;
    :param path_results: path where will be saved the results:
    :param n_partition: number of partitions;
    :param path_partition: the partition files path;
    :param SSM: name of semantic similarity measure;
    :param aspects: list of semantic aspects;
    """
    list_ents, list_ss, list_ss_baselines, list_labels = read_SS_dataset_file(path_file_SS, path_dataset_file)

    dict_ML = {}
    for algorithm in algorithms:
        dict_ML[algorithm] = []
        ensure_dir(path_results  + "/" + SSM + "/" + algorithm + "/")
        file_ML = open(path_results  + "/" + SSM + "/" + algorithm + "/" + "PerformanceResults.txt", 'w')
        file_ML.write('Run' + '\t' + 'WAF' + '\t' + 'Fmeasure(non-interact)' + '\t' + 'Fmeasure(interact)' + '\t' + 'Precision' + '\t' + 'Recall' + '\t' + 'Accuracy' + '\n')
        file_ML.close()

    n_pairs = len(list_labels)
    for Run in range(1, n_partition + 1):

        file_partition = path_partition + str(Run) + '.txt'
        test_index = process_indexes_partition(file_partition)
        train_index = list(set(range(0, n_pairs)) - set(test_index))

        print("######   RUN" + str(Run) + "       #######")

        list_labels = np.array(list_labels)
        y_train, y_test = list_labels[train_index], list_labels[test_index]
        y_train, y_test = list(y_train), list(y_test)

        list_ss = np.array(list_ss)
        list_ss_baselines = np.array(list_ss_baselines)

        X_train, X_test = list_ss[train_index], list_ss[test_index]
        X_train_baselines, X_test_baselines = list_ss_baselines[train_index], list_ss_baselines[test_index]
        X_train, X_test, X_train_baselines, X_test_baselines = list(X_train), list(X_test), list(X_train_baselines), list(X_test_baselines)

        for algorithm in algorithms:
            path_output_predictions = path_results  + '/' + SSM + "/" + algorithm + "/Predictions__" + SSM + "__" + dataset_name +  "__Run" + str(Run)

            if algorithm == "XGB":
                path_ouput_importance  = path_results + '/' + SSM + "/" + algorithm + "/FeatureImportances__" + SSM + "__" + dataset_name + "__Run" + str(Run)
                waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy  = ML.performance_XGB(X_train, X_test, y_train, y_test, path_output_predictions,
                                           path_ouput_importance, aspects)

            elif algorithm == 'RF':
                path_ouput_feature = path_results + '/' + SSM + "/" + algorithm + "/FeatureImportances__" + SSM + "__" + dataset_name + "__Run" + str(Run)
                waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy = ML.performance_RF(X_train, X_test, y_train, y_test, path_output_predictions,path_ouput_feature, aspects)

            elif algorithm.startswith('DT'):
                filename_Modeloutput = path_results + '/' + SSM + "/" + algorithm + "/Model__" + SSM + "__" + dataset_name + "__Run" + str(Run)
                if algorithm == 'DT':
                    waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy = ML.performance_DT(X_train, X_test, y_train, y_test, path_output_predictions, filename_Modeloutput, aspects)
                else:
                    depth = int(algorithm.split('DT')[1])
                    waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy  = ML.performance_DT(X_train, X_test, y_train, y_test, path_output_predictions, filename_Modeloutput, aspects, depth)

            elif algorithm.startswith('GP'):
                filename_model_gp = path_results + '/' + SSM + "/" + algorithm + "/Model__" + SSM + "__" + dataset_name + "__Run" + str(Run)
                if algorithm == 'GP':
                    waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy = ML.performance_GP(X_train, X_test, y_train, y_test, path_output_predictions, filename_model_gp)
                elif algorithm == 'GP6x':
                    waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy = ML.performance_GP_MaxDepth(X_train, X_test, y_train, y_test, path_output_predictions, filename_model_gp, "6",  ['add', 'sub', 'max', 'min'])
                else:
                    depth =algorithm.split('GP')[1]
                    waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy = ML.performance_GP_MaxDepth(X_train, X_test, y_train, y_test, path_output_predictions, filename_model_gp, depth)

            dict_ML[algorithm].append([waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy])


    file_results_ML = open(path_results + '/' + SSM + "/ResultsML.txt" , 'a')
    print('*******************')

    for algorithm in algorithms:
        print("Median " + algorithm + ": " )
        print('*******************')
        values_performance = dict_ML[algorithm]
        wafs = []
        file_algorithm =open(path_results  + '/' + SSM + "/" + algorithm + "/PerformanceResults.txt", 'a')
        for i in range(len(values_performance)):
            waf, fmeasure_noninteract, fmeasure_interact, precision, recall, accuracy = values_performance[i]
            file_algorithm.write(str(i+1) + '\t' + str(waf) + '\t' + str(fmeasure_noninteract) + '\t' + str(fmeasure_interact) + '\t' + str(precision) + '\t' + str(recall) + '\t' + str(accuracy) + '\n')
            wafs.append(waf)
        file_algorithm.close()
        file_results_ML.write(algorithm + '\t' + str(median(wafs)) + '\n')
        print(str(median(wafs)))
    file_results_ML.close()



if __name__ == "__main__":

    ###############################
    n_partition = 10

    SSMs = ["simGIC_ICSeco", "ResnikMax_ICSeco"]
    embSSMs = ['rdf2vec_skip-gram_wl', 'owl2vec_skip-gram_wl']

    algorithms = ['XGB', 'DT3', 'DT6', 'DT', 'GP', 'GP6x', 'GP3', 'RF']

    dataset = "v11"
    path_partition = 'Prediction/StratifiedPartitions/Indexes__crossvalidationTest__Run'
    path_dataset_file = "Data/" + dataset + "(score950).txt"

    type_aspects = ["roots", "subroots", "subroots_notLeave"]

    for type in type_aspects:

        type_aspects_file = open("Data/SemanticAspects_" + type + ".txt", 'r')
        n_aspects = str(type_aspects_file.readline())[:-1] + 'SAs'
        aspects=[]
        for line in type_aspects_file:
            url, name = line[:-1].split("\t")
            aspects.append(name.replace(" ", "_").replace("/", "-"))
        type_aspects_file.close()

        path_results = "Prediction/SSprov/" + n_aspects

        for SSM in SSMs:
            path_file_SS = "SS_Calculation/SS_files/" + n_aspects + '/ss_' + SSM + '.txt'
            run_cross_validation(algorithms, path_file_SS, path_dataset_file, dataset, path_results, n_partition,
                                 path_partition, SSM, aspects)
            gc.collect()

        for embSSM in embSSMs:
            path_file_SS = 'SS_Embedding_Calculation/Embedding_SS_files/embedss_200_' + embSSM + '_STRING_' + dataset + '_' + n_aspects + '.txt'
            run_cross_validation(algorithms, path_file_SS, path_dataset_file, dataset, path_results, n_partition,
                                 path_partition, embSSM, aspects)
            gc.collect()





