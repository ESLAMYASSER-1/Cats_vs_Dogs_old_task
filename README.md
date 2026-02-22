# Cats vs Dogs – Binary Image Classification (Transfer Learning)
### 20-8-2024 ----> Pytorch course (transfere learning)
### Eng: Eslam Yasser 

## Setup

```bash
pip install -r requirements.txt
```

Place your `kaggle.json` API token in `~/.kaggle/kaggle.json`
(or set `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables).

## Run

```bash
python main.py
```

The script will automatically:
1. Download and unzip the Kaggle dataset
2. Build an 80/20 train/val split
3. Run Phase 1 training (frozen backbone, head only)
4. Run Phase 2 fine-tuning (layer4 + head, differential LRs)
5. Save the best checkpoints to `checkpoints/`



