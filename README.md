# Voice-Based Gender Classification Using Pitch Analysis

Reproducible academic project using acoustic features and logistic regression.

## Run
```bash
pip install -r requirements.txt
python src/download_dataset.py
python src/train.py --data data/voice.csv --output outputs/real_voice
```
Dataset: 3,168 samples and 20 numeric features, from [primaryobjects/voice-gender](https://github.com/primaryobjects/voice-gender), CC BY-NC-SA 4.0. Data is downloaded directly rather than redistributed.

The academic report states **97.51% accuracy**, but this has **not** been independently reproduced. The model writes actual held-out accuracy and a confusion matrix to outputs. Voice features cannot reliably determine gender identity.

The original reports, presentations and notebooks are in the downloadable project archive provided in ChatGPT.