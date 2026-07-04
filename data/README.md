# Dataset

This project uses the **NetsLab-5GORAN-IDD** dataset (upper/network layer).

## Download

1. Go to [https://www.kaggle.com/datasets/netslabdemo/netslab-5g-oran-idd](https://www.kaggle.com/datasets/netslabdemo/netslab-5g-oran-idd).
2. Download `Network_Dataset.csv` (≈200 MB).
3. Place it in this `data/` folder, or upload it to Google Drive under `CTI/Network_Dataset.csv` if running on Colab.

## Dataset Details

| Property | Value |
|----------|-------|
| Rows | 1,723,817 |
| Columns | 26 |
| Target column | `attack_category` |
| Classes used | benign, ddos, dos, probe, web (bruteforce excluded — see notebook Section 3.1) |
| Source | NetsLab, University of Aveiro |

The dataset is **not included in this repository** due to file size. The notebook will load it from the path configured in the Setup cell.
