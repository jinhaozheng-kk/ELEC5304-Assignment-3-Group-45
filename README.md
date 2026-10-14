# ELEC5304 Assignment 3: PASCAL VOC 2007 Object Detection

This repository contains the Group 45 submission for ELEC5304 Assignment 3.
The project trains and evaluates an object detection model on the PASCAL VOC 2007 dataset.

## Group Members

- Jinhao Zheng
- Yilin Wu
- Huiwen Zhang
- Yiqian Guo
- Chengnan Zhu

## Project Overview

The assignment focuses on object detection using the PASCAL VOC 2007 dataset. The notebook includes:

- Dataset preparation for VOC 2007
- XML annotation parsing and bounding-box visualization
- Faster R-CNN model implementation
- Training and validation pipeline
- Test-set evaluation using mAP@0.5
- Class-wise AP analysis
- Representative error-case visualization and discussion

## Model

The detector is based on Faster R-CNN with a MobileNetV3-FPN backbone. The model is adapted for VOC by:

- Replacing the COCO prediction head with a VOC-specific head for 20 classes plus background
- Adjusting anchor sizes for small and medium VOC objects
- Fine-tuning selected backbone layers
- Using validation mAP@0.5 for checkpoint selection

## Result

The final model achieved:

```text
Test mAP@0.5: 0.7211
```

This exceeds the assignment requirement of 60% mAP@0.5.

## Files

- `Assignment_3_Group_45.ipynb`: Main notebook containing implementation, results, visualizations, and discussion.

## Environment

The notebook was developed and run in Google Colab. It uses Python, PyTorch, TorchVision, NumPy, Pandas, Matplotlib, PIL, and XML parsing utilities.

## Notes

The PASCAL VOC 2007 dataset is downloaded by the notebook and is not included in this repository.
