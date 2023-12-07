import rdflib
from rdflib.namespace import RDF, OWL, RDFS
import networkx as nx
import os

def _identity(x): return x


def _rdflib_to_networkx_graph(
        graph,
        nxgraph,
        calc_weights,
        edge_attrs,
        transform_s=_identity, transform_o=_identity):
    """Helper method for multidigraph, digraph and graph.
    Modifies nxgraph in-place!
    Arguments:
        graph: an rdflib.Graph.
        nxgraph: a networkx.Graph/DiGraph/MultiDigraph.
        calc_weights: If True adds a 'weight' attribute to each edge according
            to the count of s,p,o triples between s and o, which is meaningful
            for Graph/DiGraph.
        edge_attrs: Callable to construct edge data from s, p, o.
           'triples' attribute is handled specially to be merged.
           'weight' should not be generated if calc_weights==True.
           (see invokers below!)
        transform_s: Callable to transform node generated from s.
        transform_o: Callable to transform node generated from o.
    """
    assert callable(edge_attrs)
    assert callable(transform_s)
    assert callable(transform_o)
    import networkx as nx
    for s, p, o in graph:
        ts, to = transform_s(s), transform_o(o)  # apply possible transformations
        data = nxgraph.get_edge_data(ts, to)
        if data is None or isinstance(nxgraph, nx.MultiDiGraph):
            # no edge yet, set defaults
            data = edge_attrs(s, p, o)
            if calc_weights:
                data['weight'] = 1
            nxgraph.add_edge(ts, to, **data)
        else:
            # already have an edge, just update attributes
            if calc_weights:
                data['weight'] += 1
            if 'triples' in data:
                d = edge_attrs(s, p, o)
                data['triples'].extend(d['triples'])


def process_GAFversions(G, path_annotations_file):
    GO_classes = []
    annotations = {}
    annot_file = open(path_annotations_file, 'r')
    for line in annot_file:
        if not line.startswith("!") and line != "":
            fields = line.split("	")
            prot, GO = fields[1], fields[4]
            GO_url =  "http://purl.obolibrary.org/obo/GO_" + GO.split("GO:")[1]
            GO_classes.append(GO_url)

            descendants = nx.descendants(G, rdflib.term.URIRef(GO_url))

            if prot in annotations:
                annotations[prot].append(GO_url)
                for descendant in descendants:
                    annotations[prot].append(str(descendant))
                    GO_classes.append(str(descendant))
            else:
                annotations[prot] = [GO_url]
                for descendant in descendants:
                    annotations[prot].append(str(descendant))
                    GO_classes.append(str(descendant))
    annot_file.close()

    percentage_annotations = {}
    GO_classes = list(dict.fromkeys(GO_classes))
    for GO in GO_classes:
        count = 0
        for prot in annotations:
            if GO in annotations[prot]:
                count = count + 1
        percentage_annotations[GO] = count/len(annotations) * 100

    return percentage_annotations


def filter_classes_by_height(classes, G, beta):
    leaves = [x for x in G.nodes() if G.in_degree(x) == 0]
    return [filtered_class for filtered_class in classes if filtered_class not in leaves]


def filter_classes_by_coverage(classes, percentage_annotations, gamma):
    return [filtered_class for filtered_class in classes if percentage_annotations[str(filtered_class)]>=gamma]


def get_subclasses(g_ontology, semantic_aspects):
    new_semantic_aspects = []
    for semantic_aspect in semantic_aspects:
        for (s, p, o) in g_ontology.triples((None, rdflib.RDFS.subClassOf, rdflib.term.URIRef(semantic_aspect))):
            if not (isinstance(s, rdflib.term.BNode)):
                new_semantic_aspects.append(s)
    return new_semantic_aspects


def get_semantic_aspects(path_output_sa, path_ontology_file, path_annotations_file, alpha, gamma, beta):
    g_ontology = rdflib.Graph()
    g_ontology.parse(path_ontology_file, format='xml')

    g = rdflib.Graph()
    for (sub, pred, obj) in g_ontology.triples((None, RDFS.subClassOf, None)):
        if g_ontology.__contains__(
                (sub, rdflib.term.URIRef('http://www.geneontology.org/formats/oboInOwl#hasOBONamespace'), None)):
            if g_ontology.__contains__(
                    (obj, rdflib.term.URIRef('http://www.geneontology.org/formats/oboInOwl#hasOBONamespace'), None)):
                g.add((sub, pred, obj))
    G = nx.DiGraph()
    _rdflib_to_networkx_graph(g, G, calc_weights=False, edge_attrs=lambda s, p, o: {})

    non_roots = set()
    roots = set()
    for x, y in g_ontology.subject_objects(rdflib.RDFS.subClassOf):
        non_roots.add(x)
        if x in roots:
            roots.remove(x)
        if y not in non_roots:
            if not (isinstance(y, rdflib.term.BNode)):
                roots.add(y)
    semantic_aspects = [root for root in roots]

    if beta > 0:
        semantic_aspects = filter_classes_by_height(semantic_aspects, G, beta)
    if gamma > 0:
        percentage_annotations = process_GAFversions(G, path_annotations_file)
        semantic_aspects = filter_classes_by_coverage(semantic_aspects, percentage_annotations, gamma)

    while len(semantic_aspects) < alpha:
        semantic_aspects = get_subclasses(g_ontology, semantic_aspects)
        if beta > 0:
            semantic_aspects = filter_classes_by_height(semantic_aspects, G, beta)
        if gamma > 0:
            percentage_annotations = process_GAFversions(G, path_annotations_file)
            semantic_aspects = filter_classes_by_coverage(semantic_aspects, percentage_annotations, gamma)

    name_aspects = []
    for aspect in semantic_aspects:
        for (sub, pred, obj) in g_ontology.triples((aspect, RDFS.label, None)):
            name_aspects.append((str(aspect), str(obj)))

    str_aspects = ''
    with open(path_output_sa, 'w') as output_sa:
        for tuple in name_aspects:
            output_sa.write(tuple[0] + "\t" + tuple[1] + "\n")
            str_aspects = str_aspects + '"' + tuple[0] + '" '

    return semantic_aspects, name_aspects, str_aspects


def main(path_output, path_output_sa, path_ontology_file, path_annotations_file, path_pairs_file, alpha, gamma, beta):

    semantic_aspects, name_aspects, str_aspects = get_semantic_aspects(path_output_sa, path_ontology_file, path_annotations_file, alpha, gamma, beta)

    command_1 = 'javac -cp ".;./SS_Calculation/jar_files/*" ./SS_Calculation/Run_SS_calculation_SAs.java'
    os.system(command_1)
    command_2 = 'java -cp ".;./SS_Calculation/jar_files/*" SS_Calculation/Run_SS_calculation_SAs' + ' "' + path_ontology_file + '" "' + path_annotations_file + '" "GO" "http://purl.obolibrary.org/obo/GO_" "gaf" "' + path_pairs_file + '" "' + path_output + '" "resnik_max_ICseco" "' + str(
        len(semantic_aspects)) + '" '
    os.system(command_2 + str_aspects)

    return name_aspects

