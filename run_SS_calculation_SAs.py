import os
from process_KG import get_semantic_aspects_root, get_semantic_aspects_subroot, get_semantic_aspects_subroot_notLeave

def ensure_dir(path):
    """
    Check whether the specified path is an existing directory or not. And if is not an existing directory, it creates a new directory.
    :param path: path-like object representing a file system path;
    """
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)

def run_SS_calculation_SAs(kg_file, ontology, ontology_annotations, ontology_annotations_format,  namespace, namespace_uri, dataset, dataset_file, entity_file, path_SS, path_output_embeddings, path_embSS, type_aspects):

    if type_aspects == "roots":
        semantic_aspects, name_aspects  = get_semantic_aspects_root(ontology,ontology_annotations,kg_file, entity_file)
    if type_aspects == "subroots":
        semantic_aspects, name_aspects  = get_semantic_aspects_subroot(ontology,ontology_annotations,kg_file, entity_file)
    if type_aspects == "subroots_notLeave":
        semantic_aspects, name_aspects = get_semantic_aspects_subroot_notLeave(ontology,ontology_annotations,kg_file, entity_file)

    path_type_aspects = "Data/SemanticAspects_" + type_aspects + ".txt"
    file_semantic_aspects = open(path_type_aspects, 'w')
    file_semantic_aspects.write(str(len(semantic_aspects)) + '\n')
    str_aspects = ''
    str_name_aspects = ''
    for i in range(len(semantic_aspects)):
        str_aspects = str_aspects + '"' + semantic_aspects[i] + '" '
        aspect, name = name_aspects[i]
        str_name_aspects = str_name_aspects + '"' + name + '" '
        file_semantic_aspects.write(semantic_aspects[i] + '\t' + name_aspects[i][1] + '\n')
    file_semantic_aspects.close()

    ## Taxonomic Semantic Similarity Computation
    command_1 = 'javac -cp ".;./SS_Calculation/jar_files/*" ./SS_Calculation/Run_SS_calculation_SAs.java'
    os.system(command_1)
    path_SS_file = path_SS + str(len(name_aspects)) + "SAs/"
    ensure_dir(path_SS_file)
    command_2 = 'java -cp ".;./SS_Calculation/jar_files/*" SS_Calculation/Run_SS_calculation_SAs' + ' "' + ontology + '" "' + ontology_annotations + '" "' + namespace + '" "' + namespace_uri + '" "' + ontology_annotations_format + '" "' + dataset_file + '" "' + path_SS_file + '" "all" "' + str(len(semantic_aspects)) + '" '
    os.system(command_2 + str_aspects)

    ## Generate RDF2Vec Embeddings
    command_3 = 'python SS_Embedding_Calculation/run_RDF2VecEmbeddings.py "' + dataset + '" "' + path_output_embeddings + '" "' + kg_file + '" "' + entity_file + '" "uniform" "rdf2vec" "' + path_type_aspects + '" '
    os.system(command_3)
    ## RDF2Vec Embeddings Semantic Similarity Computation
    command_4 = 'python SS_Embedding_Calculation/run_embedSS_calculation.py "rdf2vec_skip-gram_wl" "200" "' + dataset + '" "' + dataset_file + '" "' + path_embSS + '" "' + path_output_embeddings + '" "' + path_type_aspects + '" '
    os.system(command_4)

    ## Generate OWL2Vec Embeddings
    command_5 = 'python SS_Embedding_Calculation/run_OWL2VecEmbeddings.py "' + dataset+ '" "' + entity_file + '" "uniform" "owl2vec" "' + path_output_embeddings+ '" "' + kg_file+ '" "' + path_type_aspects + '" '
    os.system(command_5)
    ## OWL2Vec Embeddings Semantic Similarity Computation
    command_6 = 'python SS_Embedding_Calculation/run_embedSS_calculation.py "owl2vec_skip-gram_wl" "200" "' + dataset + '" "' + dataset_file + '" "' + path_embSS + '" "' + path_output_embeddings + '" "' + path_type_aspects + '" '
    os.system(command_6)


if __name__ == "__main__":

    #################################################

    ontology_annotations = "Data/goa_human.gaf"
    ontology = "Data/go.owl"
    kg_file = "Data/KGs/goKG_human"
    namespace = "GO"
    namespace_uri = "http://purl.obolibrary.org/obo/GO_"
    ontology_annotations_format = "gaf"

    dataset = "STRING_v11"
    dataset_file = "Data/v11(score950).txt"
    entity_file = "Data/Prots_v11(score950).txt"

    path_output_embeddings = 'SS_Embedding_Calculation/Embeddings/'
    path_embSS = 'SS_Embedding_Calculation/Embeddings_SS_files/'
    path_SS = "SS_Calculation/SS_files/"

    semantic_aspects = ["roots", "subroots", "subroots_notLeave"]

    for type_aspect in semantic_aspects:
        run_SS_calculation_SAs(kg_file, ontology, ontology_annotations, ontology_annotations_format, namespace, namespace_uri, dataset, dataset_file, entity_file, path_SS, path_output_embeddings, path_embSS, type_aspect)

