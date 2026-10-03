# WinCLIP / WinCLIP+ on MVTec-AD (unofficial re-implementation)

Zero-shot and few-normal-shot anomaly classification & segmentation with CLIP, following
*"WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation"* (Jeong et al., CVPR 2023).
Only **MVTec-AD** is used. This is **not** the authors' code.

## Setup
```bash
git clone https://github.com/<your-username>/winclip-mvtec.git
cd winclip-mvtec
pip install -r requirements.txt
```
Download MVTec-AD yourself from https://www.mvtec.com/company/research/datasets/mvtec-ad (CC BY-NC-SA 4.0, not redistributed here).

## Run
```bash
python sanity_check.py --data_root /path/to/mvtec_anomaly_detection      # run first
python run_mvtec.py    --data_root /path/to/mvtec_anomaly_detection --shots 0 1 2 4 --seeds 5
# debugging ladder: language-only / vision-only / fused
python run_mvtec.py    --data_root ... --shots 1 --modes lang vis fused
```

## Results (MVTec-AD, mean of 15 classes; few-shot = mean of 5 seeds)
| Setting | AC AUROC | AS pAUROC | Paper AC | Paper AS |
|---|---|---|---|---|
| 0-shot WinCLIP | 90.6 | 84.3 | 91.8 | 85.1 |
| 1-shot WinCLIP+ | 92.2 | 93.3 | 93.1 | 95.2 |
| 2-shot WinCLIP+ | 93.7 | _fill_ | 93.8 | 96.0 |
| 4-shot WinCLIP+ | 94.9 | _fill_ | 94.2 | 96.2 |

## Known differences from the paper
- Prompt word lists adapted from public re-implementations (paper's supplementary lists not copied verbatim).
- Calibration used to fuse language and reference scores (per-category min-max) is my choice; the paper does not specify it.
- Zero-shot AC uses the image-scale score only (no multi-crop).

## Citation
```
@inproceedings{jeong2023winclip,
  title={WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation},
  author={Jeong, Jongheon and Zou, Yang and Kim, Taewan and Zhang, Dongqing and Ravichandran, Avinash and Dabeer, Onkar},
  booktitle={CVPR}, year={2023}}
```
