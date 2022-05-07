# Reproducibility Project for Medical Code Prediction from Clinical Notes via KSI framework

This repository is the official implementation of Reproducibility Project for Medical Code Prediction from Clinical Notes via KSI framework. 

## Original Paper
Bai, T., Vucetic, S., Improving Medical Code Prediction from Clinical Text via Incorporating Online Knowledge Sources, The Web Conference (WWW'19), 2019.


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Data

Put two files "NOTEEVENTS.csv" and "DIAGNOSES_ICD.csv" from MIMIC-III (https://mimic.physionet.org/gettingstarted/access/) under the same folder of the project.

## Preprocessing

To preprocess the data, run following three preprocessing scripts:

```preprocess1
python preprocessing1.py
```

```preprocess2
python preprocessing2.py
```

```preprocess3
python preprocessing3.py
```

## Training & Evaluation & Test

To train, eval and test the RNN, run:

```train/eval/test
python KSI_RNN.py
```

## Results

```
RNN alone:
test loss 0.034417975693941116
top- 10 0.7631632859068936
macro AUC 0.8511230834889986
micro AUC 0.9689266016943303
macro F1 0.2010727589487894
micro F1 0.6475201715285223
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
KSI+RNN:
test loss 0.03218914568424225
top- 10 0.792191104758309
macro AUC 0.8888887525103507
micro AUC 0.9759802345174364
macro F1 0.2710837962500303
micro F1 0.6605456106744171
```

## Contributing

Please contact Kehang Chang Fred (kehangc2@illinois.edu) or Haocheng Zhang (hz46@illinois.edu) for contributions.

## Developers

Haocheng Zhang
Kehang Chang Fred