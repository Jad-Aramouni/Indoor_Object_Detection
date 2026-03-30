![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-black)
![Status](https://img.shields.io/badge/Project-Completed-success)

A computer vision project for **indoor object detection** using **transfer learning** with **Ultralytics YOLO** and **GPU instead of CPU training**.

## Overview

I built a custom object detection pipeline for an indoor perception task.  
The model takes an indoor image as input and returns:

- object **class**
- object **bounding box**
- prediction **confidence**

This is more useful than plain image classification for objects, to know **where** an object is, not only **what** it is.

## Final Result

### Best model
- **Model:** YOLOv8n
- **Image size:** 640
- **Epochs:** 30
- **Hardware:** NVIDIA RTX 4060 Laptop GPU

### Best overall metrics
- **Precision:** 0.472
- **Recall:** 0.496
- **mAP50:** 0.510
- **mAP50-95:** 0.334

### Verdict
This is a **good and usable detector for this exercise**.  
It is not production-grade, but it is a strong result on a custom dataset.

---

## Dataset

The dataset was already split into:

- `train`
- `valid`
- `test`

### Classes
- `door`
- `cabinetDoor`
- `refrigeratorDoor`
- `window`
- `chair`
- `table`
- `cabinet`
- `couch`
- `openedDoor`
- `pole`

### Dataset notes
This is a **custom indoor dataset** with:
- some **class imbalance**
- some **visually similar classes**
- a few **low-frequency categories**

These factors likely explain why some classes performed much better than others.

---

## Tech Stack

- **Python**
- **PyTorch**
- **Ultralytics YOLO**
- **CUDA**
- **VS Code**
- **Windows**

### Hardware
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU

---

## What I Did

### 1. Set up the dataset
I verified the folder structure and fixed the `data.yaml` file so YOLO could correctly read:
- training images
- validation images
- test images
- labels

### 2. Built a training pipeline
I created a simple `src/` pipeline with:
- `train.py`
- `evaluate.py`
- `predict.py`
- `config.py`
- `utils.py`

### 3. Trained a pretrained detector
I used **transfer learning** with pretrained YOLO weights instead of training from scratch.

### 4. Moved from CPU to GPU
I first validated the project on CPU, then configured **CUDA-enabled PyTorch** and reran experiments on GPU.

### 5. Compared controlled experiments
I changed one major parameter at a time and compared results.

---

## Experiment Summary

| Experiment | Precision | Recall | mAP50 | mAP50-95 | Notes |
|---|---:|---:|---:|---:|---|
| Early baseline | 0.61 | 0.27 | 0.25 | 0.13 | Pipeline worked, but model was weak |
| **YOLOv8n / 30 ep / 640** | **0.472** | **0.496** | **0.510** | **0.334** | **Best overall run** |
| YOLOv8s | 0.543 | 0.448 | 0.472 | 0.310 | Better precision, worse overall quality |
| YOLOv8n / 60 ep | 0.507 | 0.444 | 0.494 | 0.326 | More epochs did not improve the best run |
| YOLOv8n / 768 imgsz | 0.492 | 0.426 | 0.469 | 0.305 | Larger image size did not help |

---

## How to Read the Metrics

### Precision
Of the objects the model predicts, how many are actually correct?

### Recall
Of the real objects in the image, how many did the model find?

### mAP50
Standard object detection metric using an IoU threshold of 0.50.

### mAP50-95
A stricter metric averaged over multiple IoU thresholds.  
It gives a better sense of localization quality.

---

## Stronger vs Weaker Classes

### Stronger classes
The model learned these classes relatively well:
- `cabinetDoor`
- `refrigeratorDoor`
- `door`
- `chair`

### Weaker classes
These remained difficult:
- `table`
- `openedDoor`
- `pole`
- `couch` *(too few examples to judge reliably)*

This suggests that **data quality and class balance** are major limiting factors.

---

## Project Structure

```text
data/
  train/
  valid/
  test/
  data.yaml

src/
  train.py
  evaluate.py
  predict.py
  config.py
  utils.py

outputs/
  runs/
  sample_predictions/
