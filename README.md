# DWLR: Domain Adaptation under Label Shift for Wearable Sensor
This repo is the implementation of paper “DWLR: Domain Adaptation under Label Shift for Wearable Sensor" accepted by IJCAI' 24.

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

## Contact
If you have any question about the code or the paper, feel free to contact me through [email](mailto:jrlee@zju.edu.cn).
