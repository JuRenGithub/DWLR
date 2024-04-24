# DWLR
This repository contains a preliminary representation of the code of DWLR. 

## Requirements
The recommended requirements for DWLR are specified as follow:
- Python 3.9
- torch==1.13.0
- numpy==1.21.5
- scikit-learn==1.0.2
- pandas==1.4.2

The dependencies can be installed by:

 ```bash
pip install -r requirements.txt
 ```

## Dataset

This repository provides the preprocessing code for [WISDM](https://www.cis.fordham.edu/wisdm/includes/files/sensorKDD-2010.pdf) dataset.

- [Download](https://www.cis.fordham.edu/wisdm/dataset.php) the WISDM dataset and unzip it.
- run ``process_data.ipynb`` to preprocess the dataset.

## Run
```bash
python main.py --freq --time
```
