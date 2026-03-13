# SNNTracker: Online High-speed Multi-Object Tracking with Spike Camera

Official implementation of our paper:

> **SNNTracker: Online High-speed Multi-Object Tracking with Spike Camera**  
> *Yajing Zheng, Chengen Li, Jiyuan Zhang, Zhaofei Yu, Tiejun Huang*  
> *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2025*  

---

## 🧠 Introduction

**SNNTracker** is a biologically inspired online multi-object tracking framework for **ultra-high-speed spike cameras**.  
It leverages **spiking neural dynamics** and **attention-based neural fields** to track multiple moving targets with millisecond latency and high robustness.

---

## ⚙️ Environment Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/YourUsername/SNNTracker.git
cd SNNTracker
pip install -r requirements.txt
```

We recommend using **Python 3.8+** and **PyTorch ≥ 1.10**.

---

## 📦 Dataset Preparation

You can download the **motVidarReal2025** dataset from the following Baidu Cloud link:

> 📁 **motVidarReal2025.zip**  
> 🔗 [https://pan.baidu.com/s/1JVDxX-adPDE3-mqSaoV-OQ?pwd=0601](https://pan.baidu.com/s/1JVDxX-adPDE3-mqSaoV-OQ?pwd=0601)  
> 🔑 Extraction Code: `0601`  
> *(Shared via Baidu Netdisk Super Member v4)*

After downloading, extract the dataset and place it under your preferred directory.  
The dataset should look like this:

```
motVidarReal2025/
├── badminton/
│   └── spikes.dat
├── cpl1/
│   └── spikes.dat
├── cplCam/
│   └── spikes.dat
├── pingpong/
│   └── spikes.dat
├── rotTrans/
│   ├── spikes.dat
│   └── spikes_gt.txt
├── spike59/
│   ├── spikes.dat
│   ├── spikes_gt.txt
└── config.yaml
```

- Each folder represents a **scene** (e.g., `spike59`, `rotTrans`, etc.).
- `spikes.dat` — Spike stream data recorded by the spike camera.  
- `spikes_gt.txt` — Ground truth annotations for object tracking.  
  - Scenes **without GT files** can only be evaluated qualitatively via visualization.  
  - Scenes **with GT files** support **quantitative evaluation** using tracking metrics.

---

## 🚀 Run the Tracker

Run the entry script `test_snntracker.py`:

```bash
python test_snntracker.py     --scene_idx 0     --attention_size 15     --data_path /root/autodl-fs/motVidarReal2020/     --label_type tracking     --metrics
```

### 🔧 Argument Description

| Argument | Short | Type | Default | Description |
|-----------|--------|------|----------|-------------|
| `--scene_idx` | `-s` | int | `0` | Index of the test scene |
| `--attention_size` | `-attn_size` | int | `15` | Size of attention window |
| `--data_path` | `-d` | str | `/root/autodl-fs/motVidarReal2020/` | Path to dataset root |
| `--label_type` | `-l` | str | `"tracking"` | Label type |
| `--metrics` | `-m` | flag | `False` | Enable quantitative metrics (requires GT) |

---

## 🖼️ Visualization and Output

For all test scenes, the script automatically saves:
- **Filtered spike frames** showing tracking trajectories.
- **Visualized motion paths** for each detected object.
- For GT-available sequences (e.g., `rotTrans`, `spike59`),  
  the code computes **quantitative metrics** such as precision and recall.

---

## 📈 Performance Summary

SNNTracker achieves real-time online tracking with **20 kHz spike streams**, maintaining high robustness under extreme motion.  
It represents one of the first demonstrations of **bio-inspired online tracking on spike cameras**.

---

## 📚 Citation

If you find this project helpful, please cite:

```bibtex
@ARTICLE{11165142,
  author={Zheng, Yajing and Li, Chengen and Zhang, Jiyuan and Yu, Zhaofei and Huang, Tiejun},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={SNNTracker: Online High-Speed Multi-Object Tracking With Spike Camera}, 
  year={2026},
  volume={48},
  number={1},
  pages={624-638},
  keywords={Cameras;Tracking;Image reconstruction;Real-time systems;Target tracking;Low latency communication;Visualization;YOLO;Neuromorphics;Lighting;DNF;high-speed MOT;online learning;STDP;spike cameras;spiking neural network;WTA},
doi={10.1109/TPAMI.2025.3610696}}
```

---

## ⚖️ License

This project is released under the **Apache 2.0 License**.  
However, **commercial use and modification without permission are strictly prohibited**.

If you reference or build upon this work, please acknowledge our paper as above.

---

## 🤝 Acknowledgment

This repository is part of our ongoing research on **neuromorphic visual perception** and **spike-based high-speed vision systems**.  
We welcome collaborations and further discussions.

---
