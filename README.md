<h1 align="center"> Explaining Protein-Protein Interactions with Knowledge Graph-based Semantic Similarity </h1>

## Pre-requesites
* install python 3.6.8;
* install java JDK 11.0.4;
* install python libraries by running the following command:  ```pip install -r req.txt```.

## Methods
KGsim2vec is a novel method to generate explainable vector representations of entity pairs in a knowledge graph to support learning with minimal losses in performance when compared to opaque models. 
This framework computes the explainable vector representations, then applies machine learning algorithms to generate predictive models, and finally generates explanations.

<img src="https://github.com/liseda-lab/ExplainablePPI/blob/main/Methodology.png"/>

## (1) Generating Explainable Features
KGsim2vec generates an explainable vector representation of entity pairs based on the semantic similarities between the entities according to different semantic aspects of the ontology, i.e., subgraphs of the ontology at the same depth.
The semantic aspects of the ontology are defined by three parameters:
* alpha is the minimum number of semantic aspects and can be set to manipulate the size and consequently the level of detail afforded by the explainable vectors (the default value is 10);
* beta is the distance to a leaf class and can be set to remove subgraphs of insufficient depth (the default value is 0);
* gamma is the percentage of entities annotated in the semantic aspects (the default value is 0);

## (2) Supervised Learning
Four types of ML algorithms are used to learn relation prediction models: decision trees (DT and DT6) and genetic programming (GP and GP6x), random forest (RF) and eXtreme gradient boosting (XGB).

## (3) Generating Explanations
For interpretable models (decision trees and genetic programming), the explanation is the model itself. However, for the black-box models (random forest and eXtreme gradient boosting), a surrogate model is added to produce local models to explain individual predictions. We employed two of the most well-known post-hoc explainability methods: LIME (Local Interpretable Model-Agnostic Explanations) and LORE (Local Rule-Based Explanations).

## (4) Evaluating Explanations
To evaluate the explanations, we considered two aspects: size and informativeness.

## Run KGsim2vec
Run the command:
```
python3 run_kgsim2vec.py path_output path_ontology_file path_annotations_file path_pairs_file alpha gamma beta
```
