import rdflib 
from rdflib.namespace import RDF, OWL, RDFS

import networkx as nx


def get_semantic_aspects_root(ontology,ontology_annotations,kg_file, entity_file):
    """
    Given an ontology, returns the semantic aspects.
    :param ontology: ontology file path in owl format;
    :return: list with semantic aspects corresponding to the subgraph roots.
    """
    g = rdflib.Graph()
    g.parse(ontology, format='xml')

    non_roots = set()
    roots = set()
    for x, y in g.subject_objects(rdflib.RDFS.subClassOf):
        non_roots.add(x)
        if x in roots:
            roots.remove(x)
        if y not in non_roots:
            if not (isinstance(y, rdflib.term.BNode)):
                roots.add(y)

    aspects = [str(root) for root in list(roots) if str(root) != "http://purl.obolibrary.org/obo/GO_0032991"]

    name_aspects = []
    for aspect in aspects:
        for (sub, pred, obj) in g.triples((rdflib.term.URIRef(aspect), RDFS.label, None)):
            name_aspect = str(obj)
            name_aspects.append((aspect, name_aspect))
            generate_GOKG_file(kg_file, ontology_annotations, g, name_aspect, aspect, entity_file)

    return aspects, name_aspects


def get_semantic_aspects_subroot(ontology,ontology_annotations,kg_file, entity_file):
    """
    Given an ontology, returns the semantic aspects.
    :param ontology: ontology file path in owl format;
    :return: list with semantic aspects corresponding to the subgraph roots.
    """

    g = rdflib.Graph()
    g.parse(ontology, format='xml')

    non_roots = set()
    roots = set()
    for x, y in g.subject_objects(rdflib.RDFS.subClassOf):
        non_roots.add(x)
        if x in roots:
            roots.remove(x)
        if y not in non_roots:
            if not (isinstance(y, rdflib.term.BNode)):
                roots.add(y)

    semantic_aspects = []
    for root in list(roots):
        for (s, p, o) in g.triples((None, rdflib.RDFS.subClassOf, root)):
            if not (isinstance(s, rdflib.term.BNode)):
                if str(s) != "http://purl.obolibrary.org/obo/GO_0032991":
                    semantic_aspects.append(str(s))

    name_aspects = []
    for aspect in semantic_aspects:
        for (sub, pred, obj) in g.triples((rdflib.term.URIRef(aspect), RDFS.label, None)):
            name_aspect = str(obj)
            name_aspects.append((aspect, name_aspect))
            generate_GOKG_file(kg_file, ontology_annotations, g, name_aspect, aspect, entity_file)

    return semantic_aspects, name_aspects


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


def get_semantic_aspects_subroot_notLeave(ontology,ontology_annotations,kg_file, entity_file):

    g_ontology = rdflib.Graph()
    g_ontology.parse(ontology, format='xml')

    non_roots = set()
    roots = set()
    for x, y in g_ontology.subject_objects(rdflib.RDFS.subClassOf):
        non_roots.add(x)
        if x in roots:
            roots.remove(x)
        if y not in non_roots:
            if not (isinstance(y, rdflib.term.BNode)):
                roots.add(y)

    g = rdflib.Graph()
    for (sub, pred, obj) in g_ontology.triples((None, RDFS.subClassOf, None)):
        if g_ontology.__contains__(
                (sub, rdflib.term.URIRef('http://www.geneontology.org/formats/oboInOwl#hasOBONamespace'), None)):
            if g_ontology.__contains__(
                    (obj, rdflib.term.URIRef('http://www.geneontology.org/formats/oboInOwl#hasOBONamespace'), None)):
                g.add((sub, pred, obj))
    G = nx.DiGraph()
    _rdflib_to_networkx_graph(g, G, calc_weights=False, edge_attrs=lambda s, p, o: {})
    leaves = [x for x in G.nodes() if G.in_degree(x) == 0]


    semantic_aspects = []
    for root in list(roots):
        for (s, p, o) in g_ontology.triples((None, rdflib.RDFS.subClassOf, root)):
            if not (isinstance(s, rdflib.term.BNode)):
                if rdflib.term.URIRef(s) not in leaves:
                    if str(s) != "http://purl.obolibrary.org/obo/GO_0032991":
                        semantic_aspects.append(str(s))

    name_aspects = []
    for aspect in semantic_aspects:
        for (sub, pred, obj) in g_ontology.triples((rdflib.term.URIRef(aspect), RDFS.label, None)):
            name_aspect = str(obj)
            name_aspects.append((aspect, name_aspect))
            generate_GOKG_file(kg_file, ontology_annotations, g_ontology, name_aspect, aspect, entity_file)

    return semantic_aspects, name_aspects


def extract_terms_per_aspect(g_ontology):
    dic_aspects, aspects = {}, set()
    g = rdflib.Graph()

    for (sub, pred, obj) in g_ontology.triples((None, RDFS.subClassOf, None)):
        if g_ontology.__contains__((sub, rdflib.term.URIRef('http://www.geneontology.org/formats/oboInOwl#hasOBONamespace'), None)):
            if g_ontology.__contains__((obj, rdflib.term.URIRef('http://www.geneontology.org/formats/oboInOwl#hasOBONamespace'), None)):
                g.add((sub, pred, obj))
                aspects.add(str(sub))
                aspects.add(str(obj))

    G = nx.DiGraph()
    _rdflib_to_networkx_graph(g, G, calc_weights=False, edge_attrs=lambda s, p, o: {})

    for aspect in aspects:
        descendants = []
        for descendant in nx.ancestors(G, rdflib.term.URIRef(aspect)):
            descendants.append(str(descendant))
        dic_aspects[aspect] = descendants + [aspect]

    return dic_aspects


def generate_GOKG_file(path_GOKG_file, annotations_file, g_ontology, aspect, url_aspect, prots_file_path):

    g_aspect = g_ontology
    GOKG_file = path_GOKG_file + "_" + aspect.replace(" ", "_").replace("/", "-") + ".owl"

    dic_aspects = extract_terms_per_aspect(g_ontology)

    file_annot = open(annotations_file, 'r')
    file_annot.readline()
    for annot in file_annot:
        list_annot = annot.split('\t')
        id_prot, GO_term = list_annot[1], list_annot[4]
        url_GO_term = "http://purl.obolibrary.org/obo/GO_" + GO_term.split(':')[1]
        url_prot = "http://www.uniprot.org/uniprot/" + id_prot

        if url_GO_term in dic_aspects[url_aspect]:
            if url_GO_term not in dic_aspects["http://purl.obolibrary.org/obo/GO_0032991"]:
                g_aspect.add((rdflib.term.URIRef(url_prot), rdflib.term.URIRef('http://www.geneontology.org/hasAnnotation'),rdflib.term.URIRef(url_GO_term)))

        if (rdflib.term.URIRef(url_prot), None, None) not in g_aspect:
            g_aspect.add((rdflib.term.URIRef(url_prot), RDF.type, rdflib.term.URIRef("http://www.w3.org/2002/07/owl#NamedIndividual")))
    file_annot.close()

    prots_file = open(prots_file_path, 'r')
    for line in prots_file:
        if (rdflib.term.URIRef(line[:-1]), None, None) not in g_aspect:
            g_aspect.add((rdflib.term.URIRef(line[:-1]), RDF.type,rdflib.term.URIRef("http://www.w3.org/2002/07/owl#NamedIndividual")))
    prots_file.close()

    g_aspect.serialize(format="application/rdf+xml", destination=GOKG_file)





