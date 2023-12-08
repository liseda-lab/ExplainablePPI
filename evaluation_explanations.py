import statistics
import rdflib
from rdflib.namespace import RDF, OWL, RDFS
import numpy as np
import pandas as pd


def process_IC(IC_calculations_file):
    dic_ics = {}
    with open(IC_calculations_file , 'r') as file_ICs:
        file_ICs.readline()
        for line in file_ICs:
            term, IC = line[:-1].split("\t")
            dic_ics[term] = float(IC)
    return dic_ics


def GOurl2GOname(ontology_file, aspects):
    dic_name2url = {}
    g_ontology = rdflib.Graph()
    g_ontology.parse(ontology_file, format='xml')
    for (sub, pred, obj) in g_ontology.triples((None, RDFS.label, None)):
        if str(sub).startswith("http://purl.obolibrary.org/obo/GO_"):
            if str(obj).replace(" ", "_").replace("-", "_") in aspects:
                dic_name2url[str(obj).replace(" ", "_").replace("-", "_")] = str(sub)
    return dic_name2url


def process_models(decisions_path_file, file_output, dic_ics, dic_name2url, aspects):
    sumICs, avgICs, n_feats = [], [], []
    with open(file_output, 'w') as output:
        output.write('Ent1\tEnt2\tSumIC\tAvgIC\tNfeats\n')
        with open(decisions_path_file, 'r') as decisions:
            for line in decisions:
                ent1, ent2, rule = line[:-1].split('\t')
                sum_IC, n_feat = 0, 0
                for aspect in aspects:
                    if aspect in rule:
                        sum_IC = sum_IC + dic_ics[dic_name2url[aspect]]
                        n_feat += 1
                if n_feat ==0:
                    print(line)
                output.write(ent1 + '\t' + ent2 + '\t' + str(sum_IC) + '\t' + str(sum_IC/n_feat) + '\t' + str(n_feat) + '\n')
                sumICs.append(sum_IC)
                avgICs.append(sum_IC/n_feat)
                n_feats.append(n_feat)
        output.write('All\tpairs\t' + str(statistics.median(sumICs)) + '\t' + str(statistics.median(avgICs)) + '\t' + str(statistics.median(n_feats)) + '\n')


def process_rule(rule, aspects, dic_ics, dic_name2url):
    sum_IC, n_feat = 0, 0
    for aspect in aspects:
        if aspect in rule:
            sum_IC = sum_IC + dic_ics[dic_name2url[aspect]]
            n_feat += 1
    return sum_IC, sum_IC/n_feat, n_feat


def process_LIMEmodels(decisions_path_file, file_output, dic_ics, dic_name2url, aspects):
    sumICs_lime3, avgICs_lime3, n_feats_lime3 = [], [], []
    sumICs_lime8, avgICs_lime8, n_feats_lime8 = [], [], []
    with open(file_output, 'w') as output:
        output.write('Ent1\tEnt2\tSumIC\tAvgIC\tNfeats\n')
        with open(decisions_path_file, 'r') as decisions:
            decisions.readline()
            for line in decisions:
                ent1, ent2, lime3, lime8 = line[:-1].split('\t')
                sum_IC_lime3, avg_IC_lime3, n_feat_lime3 = process_rule(lime3, aspects, dic_ics, dic_name2url)
                sum_IC_lime8, avg_IC_lime8, n_feat_lime8 = process_rule(lime8, aspects, dic_ics, dic_name2url)
                output.write(ent1 + '\t' + ent2 + '\t' + str(sum_IC_lime3) + '\t' + str(avg_IC_lime3) + '\t' + str(n_feat_lime3) + '\t'
                                                       + str(sum_IC_lime8) + '\t' + str(avg_IC_lime8) + '\t' + str(n_feat_lime8) + '\n')
                sumICs_lime3.append(sum_IC_lime3)
                avgICs_lime3.append(avg_IC_lime3)
                n_feats_lime3.append(n_feat_lime3)
                sumICs_lime8.append(sum_IC_lime8)
                avgICs_lime8.append(avg_IC_lime8)
                n_feats_lime8.append(n_feat_lime8)
        output.write('All\tpairs\t' + str(statistics.median(sumICs_lime3)) + '\t' + str(statistics.median(avgICs_lime3)) + '\t' + str(statistics.median(n_feats_lime3)) + '\t'
                                     + str(statistics.median(sumICs_lime8)) + '\t' + str(statistics.median(avgICs_lime8)) + '\t' + str(statistics.median(n_feats_lime8)) + '\n')


def main(path_ontology_file, IC_calculations_file, aspects, path_output, n_partition):
    
    dic_ics = process_IC(IC_calculations_file)
    dic_name2url = GOurl2GOname(path_ontology_file, aspects)

    algorithms = ['XGB', 'DT6', 'DT', 'GP', 'GP6x', 'RF']
    for run in range(1, n_partition + 1):
        for alg in algorithms:
            output_explanations =  path_output + "/" + alg + "/" + "/Explanations__" + "Run" + str(run) + ".txt"
            output_evaluation_explanations = path_output + "/" + alg + "/" + "/EvaluationExplanations__" + "Run" + str(run) + ".txt"

            if alg == "DT":
                process_models(output_explanations, output_evaluation_explanations, dic_ics, dic_name2url, aspects)

            elif alg == "DT6":
                process_models(output_explanations, output_evaluation_explanations, dic_ics, dic_name2url, aspects)

            elif alg == "GP":
                process_models(output_explanations, output_evaluation_explanations, dic_ics, dic_name2url, aspects)

            elif alg == "GP6x":
                process_models(output_explanations, output_evaluation_explanations, dic_ics, dic_name2url, aspects)

            elif alg == "RF":
                process_LIMEmodels(output_explanations, output_evaluation_explanations, dic_ics, dic_name2url, aspects)

            elif alg == "XGB":
                process_LIMEmodels(output_explanations, output_evaluation_explanations, dic_ics, dic_name2url, aspects)


