import sys
import os
import ss_calculation_sas
import supervised_learning
import evaluation_explanations

def ensure_dir(path):
    """
    Check whether the specified path is an existing directory or not. And if is not an existing directory, it creates a new directory.
    :param path: path-like object representing a file system path;
    """
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)

def main(path_output, path_ontology_file, IC_calculations_file, path_annotations_file, path_pairs_file, path_partitions_files, n_partitions, alpha, gamma, beta):
    #Generating Explainable Features
    ensure_dir(path_output + "/")
    path_output_ss = path_output + "/ss_resnik_max_ICseco.txt"
    path_output_sa = path_output + "/kgsim2vec_sas.txt"
    aspects = ss_calculation_sas.main(path_output, path_output_sa, path_ontology_file, path_annotations_file, path_pairs_file, float(alpha), float(gamma), float(beta))

    #Supervised Learning
    supervised_learning.main(path_pairs_file, path_partitions_files, path_output_ss,  aspects, path_output, n_partitions)

    #Evaluating explanations
    evaluation_explanations.main(path_ontology_file, IC_calculations_file, aspects, path_output, n_partitions)

 
if __name__ == "__main__":
    path_output = "Experimental_Results/Results/"
    path_ontology_file = "Experimental_Results/Data/go-basic.owl"
    IC_calculations_file = "Experimental_Results/Data/ICSeco_GOterms.txt"
    path_annotations_file = "Experimental_Results/Data/goa_human.gaf"
    path_pairs_file = "Experimental_Results/Data/v11(score950).txt"
    path_partitions_files = "Experimental_Results/Data/StratifiedPartitions/Indexes__crossvalidationTest__Run"
    n_partitions = 1
    main(path_output, path_ontology_file, IC_calculations_file, path_annotations_file, path_pairs_file, path_partitions_files, n_partitions, sys.argv[1], sys.argv[2], sys.argv[3])