# Execution status

- Python training pipeline previously executed with synthetic demonstration data.
- Real public voice dataset identified at https://github.com/primaryobjects/voice-gender/blob/master/voice.csv (CC BY-NC-SA 4.0).
- Real dataset training has **not** been independently executed here.
- Original report: **97.51% accuracy**, not independently reproduced.
- To run: `python src/download_dataset.py && python src/train.py --data data/voice.csv --output outputs/real_voice`.
