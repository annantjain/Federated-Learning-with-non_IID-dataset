#Label-Divergent Federated Learning with Clean & Noisy Images

## Introduction

This project explores a federated learning setup where clients have limited clean images and diverse noisy images (e.g., snow, foggy) with differing label distributions. A methodology is developed to enable clients to utilize knowledge from a central server via compressed model exchanges. The approach ensures effective labeling of clean and noisy images without pretrained models, focusing on training to handle heterogeneous client environments

## Data Loading and Processing:

data processing.ipynb code file was used for processing different hugging face datasets for different labels. These datasets were processed to two column "images" and "label". All these processed individual labelled hugging face datasets were pushed back to my personal hugging face account. The individual datasets that were processed are as follows:
### Source and Processed Datasets

Here are the source datasets and their corresponding processed datasets:

| **Source Dataset**                        | **Processed Dataset**                                      |
|-------------------------------------------|------------------------------------------------------------|
| [https://huggingface.co/datasets/Multimodal-Fatima](https://huggingface.co/datasets/Multimodal-Fatima)  | [AnnantJain/aircraft](https://huggingface.co/datasets/AnnantJain/aircraft) |
| [https://huggingface.co/datasets/StanfordCars_train](https://huggingface.co/datasets/StanfordCars_train) | [AnnantJain/cars](https://huggingface.co/datasets/AnnantJain/cars) |
| [https://huggingface.co/datasets/Mushroom_Dataset](https://huggingface.co/datasets/Mushroom_Dataset)  | [AnnantJain/mushroom](https://huggingface.co/datasets/AnnantJain/mushroom) |
| [https://huggingface.co/datasets/Rascor777](https://huggingface.co/datasets/Rascor777)  | [AnnantJain/Birds](https://huggingface.co/datasets/AnnantJain/Birds) |
| [https://huggingface.co/datasets/GATE-engine](https://huggingface.co/datasets/GATE-engine)  | [AnnantJain/vggflowers](https://huggingface.co/datasets/AnnantJain/vggflowers) |
| [https://huggingface.co/datasets/visual-layer](https://huggingface.co/datasets/visual-layer)  | [AnnantJain/oxford](https://huggingface.co/datasets/AnnantJain/oxford) |
| [https://huggingface.co/datasets/birds-image-dataset](https://huggingface.co/datasets/birds-image-dataset) | [AnnantJain/oxford](https://huggingface.co/datasets/AnnantJain/oxford) |
| [https://huggingface.co/datasets/oxford-iiit-pet-vl-enrl](https://huggingface.co/datasets/oxford-iiit-pet-vl-enrl) | [AnnantJain/pet1](https://huggingface.co/datasets/AnnantJain/pet1) |
| [https://huggingface.co/datasets/oxford-iiit-pet-vl-enrl](https://huggingface.co/datasets/oxford-iiit-pet-vl-enrl) | [AnnantJain/imagenette](https://huggingface.co/datasets/AnnantJain/imagenette) |
| [https://huggingface.co/datasets/visual-layer](https://huggingface.co/datasets/visual-layer)  | [AnnantJain/imagenette](https://huggingface.co/datasets/AnnantJain/imagenette) |
