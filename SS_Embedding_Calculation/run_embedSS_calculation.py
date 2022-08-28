import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys



def ensure_dir(path):
    """
    Check whether the specified path is an existing directory or not. And if is not an existing directory, it creates a new directory.
    :param path: A path-like object representing a file system path.
    """
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)



def process_dataset_prots(file_dataset_path):
    """
    Process the dataset file and returns a list with the proxy values for each pair of entities.
    :param file_dataset_path: dataset file path. The format of each line of the dataset files is "Ent1  Ent2   Proxy";
    :return: a list of lists. Each list represents a entity pair composed by 2 elements: a tuple (ent1,ent2) and the proxy value;
    """

    dataset = open(file_dataset_path, 'r')
    ents_proxy_list = []
    for line in dataset:
        split1 = line.split('\t')
        ent1, ent2 = split1[0], split1[1]
        proxy_value = float(split1[-1][:-1])

        url_ent1 = "http://www.uniprot.org/uniprot/" + ent1
        url_ent2 = "http://www.uniprot.org/uniprot/" + ent2

        ents_proxy_list.append([(url_ent1, url_ent2), proxy_value])
    dataset.close()
    return ents_proxy_list



def process_embedding_files(type_dataset, file_dataset_path, list_embeddings_files, output_file):
    """
    Compute cosine similarity between embeddings and write them.
    :param file_dataset_path: dataset file path with the entity pairs. The format of each line of the dataset files is "Ent1 Ent2 Proxy";
    :param list_embeddings_files: list of the embeddings files for each semantic aspect;
    :param output_file: new embedding similarity file path;
    :return: new similarity file;
    """

    if type_dataset == "Prot":
        ents_proxy_list = process_dataset_prots(file_dataset_path)

    list_dict = []
    for embedding_file in list_embeddings_files:
        dict_embeddings= eval(open(embedding_file, 'r').read())
        list_dict.append(dict_embeddings)
        
    o=open(output_file,"w")
    for pair, label in ents_proxy_list:
        ent1=pair[0]
        ent2=pair[1]
        o.write(ent1+'\t'+ent2)
        
        for dict_embeddings in list_dict:

            if (pair[0] in dict_embeddings) and (pair[1] in dict_embeddings):
                ent1 = np.array(dict_embeddings[pair[0]])
                ent1 = ent1.reshape(1,len(ent1))

                ent2 = np.array(dict_embeddings[pair[1]])
                ent2 = ent2.reshape(1,len(ent2))

                sim = cosine_similarity(ent1, ent2)[0][0]
                o.write('\t' + str(sim))
            else:
                o.write('\t' + str(0))

        o.write('\n')
    o.close()



if __name__ == "__main__":

    # ####################################

    model_embedding = sys.argv[1]
    n_embeddings = int(sys.argv[2])
    dataset = sys.argv[3]
    file_dataset_path = sys.argv[4]
    path_output =  sys.argv[5]
    path_embedding = sys.argv[6]
    path_type_aspects = sys.argv[7]

    type_aspects_file = open(path_type_aspects, 'r')
    n_aspects = str(type_aspects_file.readline())[:-1] + 'SAs'
    aspects, name_aspects = [], []
    for line in type_aspects_file:
        url, name = line[:-1].split("\t")
        aspects.append(name.replace(" ", "_").replace("/", "-"))
    type_aspects_file.close()

    list_embeddings_files = []
    for aspect in aspects:
        path_embedding_file = path_embedding + aspect + '/Embeddings_' + dataset + '_' + model_embedding + '_' + aspect + '.txt'
        list_embeddings_files.append(path_embedding_file)
    output_file = path_output + 'embedss_' + str(n_embeddings) + '_' + model_embedding + '_' + dataset + '_' +  n_aspects + '.txt'
    process_embedding_files("Prot", file_dataset_path, list_embeddings_files, output_file)


