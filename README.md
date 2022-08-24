<h1 align="center"> Prediction of Protein-Protein Interactions with Knowledge Graph-based Explainable Artificial Intelligence </h1>

## Pre-requesites
* install python 3.6.8;
* install java JDK 11.0.4;
* install python libraries by running the following command:  ```pip install -r req.txt```.

## Methods

<img src="https://github.com/liseda-lab/ExplainablePPI/blob/main/Methodology.png"/>

## (1) Computing KG-based SS for each semantic aspect

```
python3 SS_Calculation/run_SS_calculation_SAs.py
```

## (2) Training a ML algorithm and Predicting on unseen data

```
python3 Prediction/run_withPartitions.py
```
