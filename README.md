# 🛰️ KINETIC INTELLIGENCE // UNIT-V432
**Custom Trained Deep Semantic Segmentation Engine for Extreme Autonomous Navigation**

---

## 🌌 The Project Core
Unlike generic vision systems, **UNIT-V432** features a **custom-trained deep learning model** specifically optimized for unstructured terrain. By leveraging the DINOv2 (Vision Transformer) architecture as a robust feature extractor and training a custom **Segmentation Decoder** on specialized datasets, this unit achieves high-precision environment mapping.

### 🎯 Mission Objectives:
1. **Offroad Autonomous Gaming:** High-fidelity real-time detection of rocks, bushes, and passable terrain for AI-driven vehicles in simulation environments.
2. **Lunar Exploration:** A specialized calibration for **Moon Surface Navigation**, identifying craters, boulders, and lunar regolith to guide autonomous rovers through the lunar landscape.

---

## 🧠 Model Architecture & Training
* **Backbone:** DINOv2 ViT-S14 (Self-supervised pre-trained transformer).
* **Decoder:** Custom-built **SegmentationDecoder** (mapped to 10 specific terrain classes).
* **Training:** Fine-tuned on custom datasets to distinguish between critical hazards (Rocks/Boulders) and navigable paths (Landscape).
* **Inference:** Optimized for real-time performance with integrated CLAHE preprocessing for high-contrast edge detection.

---

## 🚀 Live Demo & Repository
* **Live Dashboard:** [Launch UNIT-V432 UI](https://kinetic-intelligence-v432.streamlit.app/)
* **GitHub Repository:** [Harsh28-raj/Kinetic-Intelligence-V432](https://github.com/Harsh28-raj/Kinetic-Intelligence-V432.git)

---

## 🛡️ Tactical Features
* **Custom Class Mapping:** Specifically trained to identify terrain IDs (Path, Vegetation, Rock, Sky, etc.).
* **Dynamic Trajectory Vectoring:** Calculates the center of mass of the 'Path' class to guide the navigation arrow.
* **Hazard Alert System:** Instant "CRITICAL" warnings when Rock/Boulder classes intersect the trajectory.

---

## 🛠️ Tech Stack
| Component | Technology |
| :--- | :--- |
| **Model Framework** | PyTorch (Custom Training) |
| **Feature Extractor** | Meta AI DINOv2 |
| **Data Augmentation** | Albumentations |
| **Frontend** | Streamlit (Custom Tactical CSS) |

---

## 👨‍💻 Core Developers
We are a team of AI enthusiasts dedicated to pushing the boundaries of autonomous navigation.

* **Harsh Raj** – [![GitHub](https://img.shields.io/badge/GitHub-Profile-lightgrey?style=flat&logo=github)](https://github.com/Harsh28-raj)
* **Amresh Chaurasiya** – [![GitHub](https://img.shields.io/badge/GitHub-Profile-lightgrey?style=flat&logo=github)](https://github.com/Amresh-01) 

---
