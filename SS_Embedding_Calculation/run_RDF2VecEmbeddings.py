import numpy
import random
import os
import sys
from operator import itemgetter

import rdflib
from rdflib.namespace import RDF, OWL, RDFS
import json

from pyrdf2vec.graphs import kg
from pyrdf2vec.rdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.samplers import (  # isort: skip
    ObjFreqSampler,
    ObjPredFreqSampler,
    PageRankSampler,
    PredFreqSampler,
    UniformSampler,
    RandomSampler,)
from pyrdf2vec.walkers import RandomWalker, WeisfeilerLehmanWalker


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)



########################################################################################################################
##############################################      Compute Embeddings      ############################################
########################################################################################################################

def calculate_embeddings(g, ents, path_output,  size_value, type_word2vec, n_walks , walk_depth, walker_type, sampler_type, name_embedding, domain, dataset):

    graph = kg.rdflib_to_kg(g)

    if type_word2vec == 'CBOW':
        sg_value = 0
    if type_word2vec == 'skip-gram':
        sg_value = 1

    print('----------------------------------------------------------------------------------------')
    print('Vector size: ' + str(size_value))
    print('Type Word2vec: ' + type_word2vec)

    if sampler_type.lower() == 'uniform':
        sampler = UniformSampler()
    elif sampler_type.lower() == 'predfreq':
        sampler = PredFreqSampler()
    elif sampler_type.lower() == 'objfreq':
        sampler = ObjFreqSampler()
    elif sampler_type.lower() == 'objpredfreq':
        sampler = ObjPredFreqSampler()
    elif sampler_type.lower() == 'pagerank':
        sampler = PageRankSampler()
    elif sampler_type.lower() == 'random':
        sampler = RandomSampler()

    if walker_type.lower() == 'random':
        walker = RandomWalker(depth=walk_depth, walks_per_graph=n_walks, sampler = sampler)
    elif walker_type.lower() == 'wl':
        walker = WeisfeilerLehmanWalker(depth=walk_depth, walks_per_graph=n_walks, sampler = sampler)
    elif walker_type.lower() == 'anonymous':
        walker = AnonymousWalker(depth=walk_depth, walks_per_graph=n_walks, sampler = sampler)
    elif walker_type.lower() == 'halk':
        walker = HalkWalker(depth=walk_depth, walks_per_graph=n_walks, sampler = sampler)
    elif walker_type.lower() == 'ngram':
        walker = NGramWalker(depth=walk_depth, walks_per_graph=n_walks, sampler = sampler)
    elif walker_type.lower() == 'walklet':
        walker = WalkletWalker(depth=walk_depth, walks_per_graph=n_walks, sampler = sampler)

    transformer = RDF2VecTransformer(Word2Vec(size=size_value, sg=sg_value), walkers=[walker])
    embeddings = transformer.fit_transform(graph, ents)
    with open(path_output + 'Embeddings_' + dataset + '_'+ name_embedding + '_' + str(
            type_word2vec) + '_' + walker_type + '_' +domain + '.txt', 'w') as file:
        file.write("{")
        first = False
        for i in range(len(ents)):
            if first:
                file.write(", '%s':%s" % (str(ents[i]), str(embeddings[i].tolist())))
            else:
                file.write("'%s':%s" % (str(ents[i]), str(embeddings[i].tolist())))
                first = True
            file.flush()
        file.write("}")



########################################################################################################################
##############################################        Call Embeddings       ############################################
########################################################################################################################

def run_embedddings_aspect(path_kgs, vector_size, type_word2vec, n_walks, walk_depth, walker_type, sampler_type , name_embedding, path_output, path_entity_file,  aspects, name_aspects, dataset):

    ents = [line.strip() for line in open(path_entity_file).readlines()]

    for i in range(len(aspects)):
        aspect = aspects[i]
        name_aspect = name_aspects[i]

        kg_file = path_kgs + "_" + name_aspect + ".owl"
        g = rdflib.Graph()
        g.parse(kg_file, format='xml')

        path_output_dataset = path_output + name_aspect + '/'
        ensure_dir(path_output_dataset)

        calculate_embeddings(g, ents, path_output_dataset, vector_size, type_word2vec, n_walks, walk_depth, walker_type, sampler_type, name_embedding, name_aspect, dataset)


if __name__== '__main__':

    #################################### Parameters ####################################
    vector_size = 200
    n_walks = 100
    type_word2vec = "skip-gram"
    walk_depth = 4
    walker_type = 'wl'

    ####################################
    dataset = sys.argv[1]
    path_output = sys.argv[2]
    path_kgs = sys.argv[3]
    path_entity_file = sys.argv[4]
    sampler_type = sys.argv[5]
    name_embedding = sys.argv[6]
    path_type_aspects = sys.argv[7]

    type_aspects_file = open(path_type_aspects, 'r')
    n_aspects = str(type_aspects_file.readline())[:-1] + 'SAs'
    aspects, name_aspects = [], []
    for line in type_aspects_file:
        url, name = line[:-1].split("\t")
        name_aspects.append(name.replace(" ", "_").replace("/", "-"))
        aspects.append(url)
    type_aspects_file.close()

    ensure_dir(path_output)
    run_embedddings_aspect(path_kgs, vector_size, type_word2vec, n_walks, walk_depth,
                                 walker_type, sampler_type , name_embedding, path_output, path_entity_file, aspects, name_aspects, dataset)
