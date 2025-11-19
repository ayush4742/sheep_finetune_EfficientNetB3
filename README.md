
Sheep Breed Classification using EfficientNetB3
This Google Colab notebook demonstrates a deep learning approach for classifying different breeds of sheep using transfer learning with the EfficientNetB3 architecture.

Project Goal
The primary objective of this project is to build and train an image classification model capable of accurately identifying various sheep breeds from images. This can be useful for automated livestock management, breed research, and agricultural applications.

Model Architecture
The model leverages EfficientNetB3, a highly efficient and accurate convolutional neural network, as its backbone for feature extraction. We employ a transfer learning strategy:

Base Model: EfficientNetB3, pre-trained on the ImageNet dataset, is used without its original top classification layer.
Custom Head: A custom classification head is added on top of the EfficientNetB3 base. This head consists of:
GlobalAveragePooling2D: Reduces spatial dimensions.
BatchNormalization: Stabilizes training.
Dropout layers (with rates 0.4 and 0.3): Prevents overfitting.
Two Dense layers: A hidden layer with 512 units and ReLU activation, and a final output layer with num_classes units and softmax activation for multi-class classification.
Training Strategy
The training is conducted in two distinct phases to optimize performance and prevent catastrophic forgetting:

Phase A: Head Training (Frozen Base)

The weights of the EfficientNetB3 base model are initially frozen (set to trainable=False).
Only the custom classification head's layers are trained for a few epochs (EPOCHS_HEAD). This allows the newly added layers to learn to interpret the high-level features extracted by the pre-trained EfficientNetB3.
Phase B: Fine-tuning (Unfrozen Base)

After Phase A, a portion of the EfficientNetB3 base model (specifically, the top 40 layers) is unfrozen (set to trainable=True).

# Sheep Breed Classification using EfficientNetB3

This Google Colab notebook demonstrates a deep learning approach for classifying different breeds of sheep using transfer learning with the EfficientNetB3 architecture.

## Project Goal

The primary objective of this project is to build and train an image classification model capable of accurately identifying various sheep breeds from images. This can be useful for automated livestock management, breed research, and agricultural applications.

## Model Architecture

The model leverages **EfficientNetB3**, a highly efficient and accurate convolutional neural network, as its backbone for feature extraction. We employ a transfer learning strategy:

*   **Base Model:** EfficientNetB3, pre-trained on the ImageNet dataset, is used without its original top classification layer.
*   **Custom Head:** A custom classification head is added on top of the EfficientNetB3 base. This head consists of:
    *   `GlobalAveragePooling2D`: Reduces spatial dimensions.
    *   `BatchNormalization`: Stabilizes training.
    *   `Dropout` layers (with rates 0.4 and 0.3): Prevents overfitting.
    *   Two `Dense` layers: A hidden layer with 512 units and ReLU activation, and a final output layer with `num_classes` units and `softmax` activation for multi-class classification.

## Training Strategy

The training is conducted in two distinct phases to optimize performance and prevent catastrophic forgetting:

1.  **Phase A: Head Training (Frozen Base)**
    *   The weights of the EfficientNetB3 base model are initially **frozen** (set to `trainable=False`).
    *   Only the custom classification head's layers are trained for a few epochs (`EPOCHS_HEAD`). This allows the newly added layers to learn to interpret the high-level features extracted by the pre-trained EfficientNetB3.

2.  **Phase B: Fine-tuning (Unfrozen Base)**
    *   After Phase A, a portion of the EfficientNetB3 base model (specifically, the top 40 layers) is **unfrozen** (set to `trainable=True`).
    *   The entire model (frozen lower layers of EfficientNetB3, unfrozen top layers of EfficientNetB3, and the custom head) is then fine-tuned with a significantly lower learning rate (`1e-5` vs `1e-4` in Phase A) for more epochs (`EPOCHS_FINE`). This step allows the model to adapt the pre-trained features more specifically to the sheep breed dataset.

## Dataset

The model is trained on a dataset of sheep images, structured into directories representing different breeds. The notebook assumes the data is organized with subdirectories for `train` and `val` (validation) containing breed-specific folders. The identified classes are:

*   `Marino`
*   `Poll Dorset`
*   `Suffolk`
*   `White Suffolk`

## Data Augmentation

To improve the model's generalization capabilities and robustness, on-the-fly data augmentation is applied during training. This includes:

*   `RandomFlip("horizontal")`
*   `RandomRotation(0.15)`
*   `RandomZoom(0.1)`
*   `RandomTranslation(0.05, 0.05)`

## Training Configuration

*   **Image Size:** `300x300` pixels (`IMG_SIZE`).
*   **Batch Size:** `32` (`BATCH_SIZE`).
*   **Optimizer:** Adam optimizer.
*   **Loss Function:** `sparse_categorical_crossentropy` (suitable for integer labels).
*   **Metrics:** `accuracy`.
*   **Callbacks:**
    *   `ReduceLROnPlateau`: Reduces the learning rate when the validation loss plateaus.
    *   `EarlyStopping`: Stops training if the validation loss does not improve for a specified number of epochs, restoring the best model weights.
*   **Class Weights:** Applied to address potential class imbalance in the dataset.

## Usage

To run this notebook and train your own sheep breed classifier:

1.  **Open in Google Colab:** Upload and open this notebook in Google Colab.
2.  **Mount Google Drive:** Ensure your Google Drive is mounted. The notebook will attempt to do this automatically. Your dataset should be located at `/content/drive/MyDrive/sheep` (or adjust the `DATA_PATH` variable in the code).
    *   The `sheep` directory should directly contain subdirectories for each breed (e.g., `sheep/Marino`, `sheep/Poll Dorset`, etc.).
3.  **Run Cells Sequentially:** Execute all code cells in order.

## Current Status

*   **Dataset Preparation:** The notebook correctly identifies 4 classes: `['Marino', 'Poll Dorset', 'Suffolk', 'White Suffolk']`.
*   **Phase A Training (Head Training):** Successfully completed for 8 epochs. The model learns initial weights for the custom classification head while the EfficientNetB3 base is frozen.
*   **Phase B Training (Fine-tuning):** The model is recompiled, and the top 40 layers of EfficientNetB3 are unfrozen for fine-tuning, allowing the entire model to adapt further to the specific task.
*   **Evaluation:** The notebook includes cells for evaluating the model's performance on the validation set, including a classification report and confusion matrix.
*   **Test-Time Augmentation (TTA):** A function `tta_predict` is implemented to demonstrate how Test-Time Augmentation can be used to potentially boost prediction accuracy by averaging predictions from augmented versions of a single image.
*   **Model Saving:** The final trained model is saved in the `.keras` format to your Google Drive.

## Potential Improvements and Future Work

*   **Hyperparameter Tuning:** Experiment with different learning rates, dropout values, and the number of unfrozen layers during fine-tuning.
*   **Advanced Data Augmentation:** Explore more sophisticated augmentation techniques.
*   **Cross-Validation:** Implement k-fold cross-validation for a more robust evaluation of model performance.
*   **Deployment:** Convert the model to TensorFlow Lite for deployment on mobile or edge devices.
*   **User Interface:** Implement a more comprehensive Gradio interface or a web application for easier interaction and real-time inference.
