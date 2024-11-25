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

### Dataset Merging and Client Dataset Creation

The `data_merging.ipynb` code file is used to create datasets for five clients, each with different labels and label distributions. In this code, we have used individual processed datasets from Hugging Face and merged them according to the label proportions for each client.

### Steps involved:
1. **Loading Processed Datasets**: The processed datasets from Hugging Face are loaded one by one.
2. **Creating Client Datasets**: Each client dataset is created by concatenating the required datasets. For each client, the dataset is further modified by adding noisy data (either Gaussian noise or image blur) to a specific proportion of images.
3. **Train and Test Splits**: After creating the client datasets, we split them into training and testing sets.
4. **Uploading to Hugging Face**: Finally, the federated client datasets are pushed back to my Hugging Face account for further use.

The final processed datasets are stored on Hugging Face and are organized by client. Each dataset consists of two columns:
1. **Image**: Contains image data of type `image`.
2. **Label**: Contains string labels corresponding to each image.

Here are the links to the final client datasets in my hugging face account:

- [Client 1 Dataset](https://huggingface.co/datasets/AnnantJain/client1_federated_dataset)
- [Client 2 Dataset](https://huggingface.co/datasets/AnnantJain/client2_federated_dataset)
- [Client 3 Dataset](https://huggingface.co/datasets/AnnantJain/client3_federated_dataset)
- [Client 4 Dataset](https://huggingface.co/datasets/AnnantJain/client4_federated_dataset)
- [Client 5 Dataset](https://huggingface.co/datasets/AnnantJain/client5_federated_dataset)

So you can directly use this datasets from loading it directly from my private hugging face account. No need to do any data creation and processing part.

### Experiment Overview

Before training, we apply a series of transformations to each client’s dataset to ensure uniformity in image dimensions and channel properties. As the collected images vary significantly in pixel size and color channels, these transformations standardize the dataset and prepare it for effective processing in the neural network. The transformations applied are as follows:

- **Grayscale to RGB Conversion**: Some images in the datasets are grayscale. The transformation `Grayscale(num_output_channels=3)` ensures that all images have three color channels (RGB) by converting grayscale images to a 3-channel format. This step is necessary to maintain consistent input dimensions for the model.
  
- **Resizing to 256x256 Pixels**: Images from different datasets have varying resolutions. To standardize the input size, `Resize((256, 256))` is applied, resizing all images to 256x256 pixels. This uniform size allows the model to process images efficiently and ensures compatibility across all datasets.
  
- **Conversion to Tensors**: The `ToTensor()` transformation converts images from PIL format to PyTorch tensors, which are compatible with deep learning models. This step is essential for handling images within PyTorch’s processing pipeline.
  
- **Normalization**: To scale the pixel values, the transformation `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` standardizes each RGB channel. This normalization helps stabilize the model training process by bringing the input values into a similar range, based on common means and standard deviations used in pre-trained image classification models.

### Local CNN Architecture

The local CNN architecture remains the same for all clients, ensuring uniformity in model structure across experiments. ![Architecture Image](https://github.com/annantjain/Federated-Learning-with-non_IID-dataset/blob/master/NN.png?raw=true)

### Common Training Configuration

For all experiments, the following configuration was used:
- **Total Rounds**: 5
- **Local Epochs**: 2
- **Batch Size**: 16

---

### Experiments

#### Experiment 1(a): Standard Federated Averaging (FedAvg)
- **Code File**: `FedAvg.ipynb`
- **Description**: Standard Federated Averaging algorithm.
- **Configuration**: 
  - Loss Function = Cross-Entropy
  - Learning Rate = 0.001
  - Optimizer = Adam
  - Local Epochs = 2
  - Rounds = 5
  - Batch Size = 16

#### Experiment 1(b): Federated Averaging with Weighted Aggregation
- **Code File**: `FedAvg_new.ipynb`
- **Description**: Standard Federated Averaging with weighted aggregation (where weights are determined by the noise fraction in the client dataset).
- **Configuration**: Same as Experiment 1(a).

#### Experiment 2: FedAvg with Focal Loss Regularization
- **Code File**: `FedAvg_Focal_Loss.ipynb`
- **Description**: FedAvg with Focal Loss Regularization to handle class imbalance and noisy data.
- **Focal Loss**: This modification of cross-entropy loss focuses on hard (misclassified) examples, which are typically noisy data points.
- **Configuration**: 
  - Loss Function = Focal Loss
  - Rounds = 5
  - Local Epochs = 2
  - Batch Size = 16
  - Learning Rate = 0.001
  - Gamma = 2
  - Alpha = 0.25

#### Experiment 3: FedAvg with Label Smoothing Loss Regularization
- **Code File**: `FedAvg_with_Label_Smoothing.ipynb`
- **Description**: FedAvg with Label Smoothing Loss Regularization to reduce model overconfidence in noisy labels.
- **Label Smoothing**: This technique assigns a small probability to incorrect classes, encouraging better generalization.
- **Configuration**: 
  - Loss Function = Label Smoothing
  - Rounds = 5
  - Local Epochs = 2
  - Batch Size = 16
  - Learning Rate = 0.001
  - Smoothing = 0.1, 0.2 (two experiments conducted with different smoothing factors)

#### Experiment 4: Federated Proximal (FedProx)
- **Code File**: `FedProx.ipynb`
- **Description**: FedProx is an extension of FedAvg designed to handle data heterogeneity (non-IID) by introducing a proximal term to the loss function.
- **Configuration**: 
  - Loss Function = Cross-Entropy
  - Proximal Term
  - Learning Rate = 0.001
  - Local Epochs = 2
  - Rounds = 5
  - Batch Size = 16

#### Experiment 5: Krum and Multi-Krum Aggregation Algorithms
- **Code File**: `Krum_and_Multi_krum.ipynb`
- **Description**: Krum and Multi-Krum algorithms provide Byzantine resilience by selecting reliable client updates in the presence of malicious or noisy clients.
- **Krum**: Selects the client model with the smallest sum of distances to its closest `n - f - 2` neighbors, where `f` is the number of potential Byzantine clients.
- **Multi-Krum**: Extends Krum by selecting multiple models with the lowest scores and averaging them.
- **Configuration**: 
  - Rounds = 3
  - Local Epochs = 2
  - Batch Size = 16
  - Learning Rate = 0.001

#### Experiment 6: Personalized Client Clustering
- **Code File**: `Fed_clusterPer.ipynb`
- **Description**: Personalized Client Clustering groups clients with similar data distributions and trains separate models for each cluster to improve model performance on non-IID data.
- **Configuration**: 
  - Loss Function = Cross-Entropy
  - Local Epochs = 2
  - Rounds = 5
  - Batch Size = 16

All the experiment from 7 to 10 are done in same code file: FedPer modified( final new algo).ipynb

#### Experiment 7: Personalized Federated Learning with Weighted Averaging
- **Description**: Personalized Federated Learning with weighted averaging, layer freezing, and weight pruning to improve client model accuracy and efficiency.
- **Configuration**: 
  - Loss Function = Cross-Entropy
  - Local Epochs = 2
  - Rounds = 5
  - Batch Size = 16
  - Layer Freezing
  - Pruning = 30%

#### Experiment 8: Adaptive Learning Rates and Weight Decay
- **Description**: Adaptive learning rates and weight decay to enhance model convergence and reduce overfitting.
- **Configuration**: 
  - Loss Function = Cross-Entropy
  - Local Epochs = 2
  - Rounds = 5
  - Batch Size = 16
  - Adaptive Learning Rate
  - Weight Decay = 1e-4

#### Experiment 9: Model Weight Regularization with Knowledge Distillation
- **Code File**: `FedPer_modified.ipynb`
- **Description**: Knowledge distillation where the global model serves as a teacher to refine client models, improving generalization.
- **Distillation Loss**: Defined as follows:
  
  ```python
  def distillation_loss(student_outputs, teacher_outputs, labels, alpha=0.5, T=2):
      soft_loss = nn.KLDivLoss()(nn.functional.log_softmax(student_outputs / T, dim=1),
                                  nn.functional.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T)
- **Configuration**:
  -Loss Function = Distillation Loss
  -Local Epochs = 2
  -Rounds = 5
  -Batch Size = 16
  -Alpha = 0.5
  -Temperature = 2

#### Experiment 10:  Final New Algorithm
- **Description**:My Final new algorithm (after observing results from all above experiments)
In this experiment, label smoothing was combined with personalized federated learning to improve model resilience to label noise across clients. To ease computational load, we applied of freezing of higher layers after a particular round ensuring that the updation is only happening in lower layers ( fully connected layers). In addition to it, pruning of weights to zero was ensured if its importance is less than the threshold (30%). The evaluation accuracy of each client is used as a weight for that client in weighted averaging to give a global model. Now this algorithm has compact parameter exchanges with less load and comparatively better accuracy values.
- **Configuration**:
  -Loss Function = Label smoothing (smoothing = 0.1)
  -Local Epochs = 2
  -Rounds = 5
  -Batch Size = 16

### Novelty Experiments

#### Experiment 11: Final New Algorithm with Adaptive Pruning Methods
- **Code File**: `Final new algo with adaptive pruning-novelty.ipynb`
- **Description**: Implementation of adaptive pruning methods in the final algorithm to improve model performance and reduce unnecessary complexity. But didn't observe the boost in accuracy %, obtained comparable results with my Exp 10.

#### Experiment 12: Final New Algorithm with a Newly Developed Loss Function
- **Code File**: `Final new algo with new Loss Function-novelty.ipynb`
- **Description**: Introduction of a newly developed loss function to enhance model performance in federated learning.
#### New Loss Function:
This loss function designed to enhance the performance of personalized federated learning by combining several optimization strategies.

#### **Key Components of FDAL**
1. **Classification with Noise Scaling**:  
   Focuses on accurate classification, weighted by \( (1 - \eta) \), giving more importance to clean data for clients with higher noise ratios.

2. **Federated Alignment**:  
   Uses \( \text{KLDiv}(\mathbf{p} \parallel \mathbf{p}_\text{global}) \) to align local predictions with the global model while allowing client-specific variations.

3. **Entropy Regularization**:  
   Penalizes overconfident predictions to encourage caution, particularly in noisy environments.

4. **Confidence-Calibrated Alignment**:  
   Assigns a higher penalty to incorrect predictions made with high confidence, reducing the likelihood of overfitting on noisy data.

#### **Mathematical Expression**
The loss function is formulated as:
\[
\mathcal{L} = (1 - \eta) \cdot \text{CE}(\mathbf{p}(x_i), y_i) 
+ \alpha \cdot \text{KLDiv}(\mathbf{p}(x_i) \parallel \mathbf{p}_\text{global}(x_i)) 
+ \beta \cdot \mathcal{H}(\mathbf{p}(x_i)) 
+ \gamma \cdot (1 - c(x_i)) \cdot \|\mathbf{p}(x_i) - \mathbf{p}_\text{global}(x_i)\|_2^2
\]

#### **Notations**
- \( \mathbf{p}(x_i) \): Prediction probabilities of the local model for input \( x_i \).
- \( \mathbf{p}_\text{global}(x_i) \): Global model predictions for \( x_i \).
- \( y_i \): True label.
- \( \eta \): Noise ratio for the current client.
- \( c(x_i) = \max(\mathbf{p}(x_i)) \): Confidence score of the prediction.
- \( \alpha, \beta, \gamma \): Hyperparameters controlling the influence of each term in the loss.

  This loss Function has increased our accuracy % of 4 clients little bit. Hence this is the better loss function then Label smoothing loss function.

### Interactive User Interface
- **Code File**: `UI.ipynb`
  
In this experiment, we have used the `gradio` library to create an interactive user interface that evaluates the performance of 5 trained and saved client models. The interface accepts an input batch of images, which are named in the format `label_index.jpg`. The actual label of each image is extracted from the name before the underscore (`_`). 

For example, if an input image is named `cat_1.jpg`, the label "cat" is extracted from the filename before the underscore.

For each client model, the predicted labels are compared against the true labels. The output of the UI displays the percentage of correctly labeled images by each client model. Additionally, the interface provides an analysis showing which images were labeled correctly and which ones were predicted incorrectly.

This interface allows easy evaluation and comparison of client models' performance on the provided dataset.
 
