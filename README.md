# 🍴 MLFork: Bearing Fault Diagnosis via a Mamba-Powered Few-Shot Learning Model with Multi-Level Architecture and Local Vector Attention

📄 **Paper:** https://doi.org/10.1016/j.neucom.2025.131518
        
        

---

## 🧠 Model Overview

### 1. Main Architecture
![Main Model](images/model.png)

### 2. Feature Extractor Modules
![BCVSS](images/bcvss.png)
![SAFE](images/safe.png)

### 3. Vector Attention Modules
![PSVA](images/psva.png)
![PCVA](images/pcva.png)

---

## ⚙️ Requirements

- Python ≥ 3.8  
- Linux OS  
- PyTorch ≥ 0.4  
- NVIDIA GPU with CUDA + cuDNN  

---

## 📊 Datasets

This work evaluates performance on two benchmark bearing fault datasets:

- **CWRU** — Case Western Reserve University Bearing Dataset  
  https://engineering.case.edu/bearingdatacenter  

- **PU** — Paderborn University Bearing Dataset  
  https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter/data-sets-and-download  

---

## 🚀 Getting Started

### Clone Repository
```bash
git clone https://github.com/thzhere/MLFork.git
cd MLFork

## Getting Started
- Installation
``` bash
git clone https://github.com/thzhere/MLFork.git
```
- Training for 1 shot
``` bash
python train_1shot.py --dataset 'CWRU' --training_samples_CWRU 30 --training_samples_PDB 195 --model_name 'Net'
```
- Testing for 1 shot
```bash
python test_1shot.py --dataset 'CWRU' --best_weight 'PATH TO BEST WEIGHT'
```
- Training for 5 shot
``` bash
python train_5shot.py --dataset 'CWRU' --training_samples_CWRU 60 --training_samples_PDB 300 --model_name 'Net'
```
- Testing for 5 shot
```bash
python test_5shot.py --dataset 'CWRU' --best_weight 'PATH TO BEST WEIGHT'
```
- Result
1. CWRU dataset
![plot](images/cwru.png)
2. PU dataset
![plot](images/pu.png)

## Contact
Please feel free to contact me via email thai.nd232543@sis.hust.edu.vn or duythainsl@gmail.com if you need anything related to this repo!
## Citation
If you feel this code is useful, please give us 1 ⭐ and cite our paper.
```bash
@article{nguyen2025mlfork,
  title={MLFork: Bearing fault diagnosis via Mamba-powered few-shot learning model with multi-level architecture enhanced by spatial-wise and channel-wise local vector attention},
  author={Nguyen, Duy-Thai and Nguyen, Van-Quoc-Viet and  Tran, Thi-Thao and Pham, Van-Truong},
  journal={Neurocomputing},
  pages={131518},
  year={2025},
  publisher={Elsevier}
}
```
