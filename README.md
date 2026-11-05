# CIFAR-10 Image Classification with ResNet50

This project applies transfer learning to classify CIFAR-10 RGB images with an ImageNet-pretrained ResNet50 convolutional neural network.

The complete implementation is available in the [CIFAR-10 ResNet50 notebook](./notebooks/cifar10_resnet50_transfer_learning.ipynb). The final run was executed in Google Colab on a GPU.

## Project objective

Build and evaluate an image-classification pipeline that:

1. Loads the CIFAR-10 dataset.
2. Uses a balanced sample of 10,000 training images, as required by the project brief.
3. Preprocesses the images for ResNet50.
4. Trains a custom classification head while the ResNet50 base is frozen.
5. Unfreezes and fine-tunes the complete model.
6. Evaluates the final model on the complete 10,000-image test set.
7. Documents the results, trade-offs, limitations, and possible improvements.

## Dataset

CIFAR-10 contains 60,000 colour images with a resolution of 32x32 pixels:

- 50,000 original training images
- 10,000 test images
- 10 mutually exclusive classes
- 6,000 images per class

The classes are:

`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, and `truck`.

This implementation samples 10,000 training images stratified by class. The test set is not reduced.

## Model and workflow

The notebook uses the following pipeline:

```text
CIFAR-10
  -> balanced 10,000-image training sample
  -> ResNet50 application preprocessing
  -> 8,000-image fit set + 2,000-image validation set
  -> frozen ResNet50 convolutional base
  -> global average pooling and custom dense classification head
  -> head training for up to 10 epochs
  -> full-model fine-tuning for up to 10 epochs
  -> test evaluation and confusion matrix
```

### Preprocessing

The notebook uses `tensorflow.keras.applications.resnet50.preprocess_input`, which applies the preprocessing expected by the ImageNet-trained ResNet50 weights.

CIFAR-10 images already have the required `(32, 32, 3)` input shape, so they are not resized. The training data is split before model fitting, with 200 validation images per class, so that validation images remain separate from the training batches.

Light data augmentation (horizontal flips and small shifts) runs inside the model during training only. An optional `IMAGE_SIZE` setting can upsample images inside the model. It defaults to 32 to match the brief; 96 is usually much more accurate but about nine times slower.

### Transfer learning stages

**Stage 1: frozen base**

- ResNet50 weights are initialized from ImageNet.
- The convolutional base is frozen.
- Only the custom classification head is trained.
- Learning rate: `1e-3`.
- Maximum duration: 10 epochs.

**Stage 2: fine-tuning**

- The ResNet50 base is unfrozen, except for its Batch Normalization layers, which keep their ImageNet statistics.
- The complete model is trained with a smaller learning rate.
- Learning rate: `1e-5`.
- Maximum duration: 10 epochs.

Early stopping and learning-rate reduction callbacks are included to reduce unnecessary training when validation loss stops improving. New callback objects are created for each stage, because Keras 3 `EarlyStopping` keeps its best value across `fit()` calls.

## Running the project

### Recommended environment

Run the notebook in [Google Colab](https://colab.research.google.com/) with a GPU runtime.

The fine-tuning stage can take a long time on a CPU. To enable a GPU in Colab:

1. Open **Runtime**.
2. Select **Change runtime type**.
3. Choose **T4 GPU** or another available GPU.
4. Save the setting.

### Execution steps

1. Open [cifar10_resnet50_transfer_learning.ipynb](./notebooks/cifar10_resnet50_transfer_learning.ipynb) in Google Colab.
2. Run the cells from top to bottom.
3. Wait for CIFAR-10 and the ResNet50 ImageNet weights to download.
4. Review the model summary and training histories.
5. Record the final test loss and accuracy.
6. Inspect the confusion matrix and per-class metrics.
7. If you rerun the notebook, update the "Ergebnisse dieses Durchlaufs" (results of this run) part of the analysis section with the printed values or `results/metrics.json`. Results differ slightly between runs and hardware.

Plots and metrics are saved to `results/` (in Colab, `/content/results/`; download them from there).

The notebook installs no custom packages. It requires:

- Python 3
- TensorFlow
- NumPy
- Matplotlib

## Expected deliverables

Before submission, the executed notebook should contain:

- Dataset loading and balanced sampling logic
- Image inspection and preprocessing
- ResNet50 architecture and custom classification head
- Evidence that the base is frozen during the first stage
- Evidence that the base is unfrozen during fine-tuning
- Training and validation histories
- Final test loss and accuracy
- Confusion matrix
- Class-level accuracy metrics and exported result JSON
- Written analysis of:
  - preprocessing decisions
  - differences between the two training stages
  - performance trends and overfitting
  - data, runtime, and hardware limitations
  - possible next steps

## Current run status

The final notebook has been fully executed in Google Colab on a GPU (10 + 10 epochs), and its analysis is complete.

| Metric | First run | Final run |
| --- | --- | --- |
| Head stage best validation accuracy | 62.1% | 64.55% |
| Fine-tuning best validation accuracy | 48.1% | ≈ 71% |
| Fine-tuning epochs run | 4 | 10 |
| Test loss | 1.5373 | 0.9039 |
| Test accuracy | 47.00% | **70.88%** |
| Macro-averaged class accuracy | – | 70.88% |

- Strongest classes: automobile (86.6%), frog (86.1%), airplane and horse (78.1% each)
- Weakest classes: cat (50.8%), deer (59.1%), bird (59.5%)
- Most confused pairs: cat → dog (140), dog → cat (140), truck → automobile (132), cat → frog (118), ship → airplane (116)
- Frog has one of the highest recalls (86.1%) but only 67.1% precision: the model over-predicts "frog" for cats, birds, and deer.
- Overfitting: moderate after fine-tuning, with about 83% training vs. 71% validation accuracy.

The plots and metrics of this run are in [`results/`](./results/).

Notes on reading these numbers:

- **Macro accuracy equals test accuracy** because the test set has exactly 1,000 images per class.
- **The tested model is the epoch-10 model.** The lowest fine-tuning validation *loss* was at epoch 7 (0.9119, 69.85% accuracy), which is why `metrics.json` reports `best_epoch: 7`. Early stopping never triggered (patience 4, only 3 epochs without improvement), so `restore_best_weights` did not restore anything.
- **Results vary slightly between runs.** GPU training is not fully deterministic even with a fixed seed. Earlier runs of the same code reached 70.97% (Colab GPU) and 72.00% (local CPU).

The first run reached:

- Head stage best validation accuracy: 62.1% (epoch 4)
- Final test loss: `1.5373`
- Final test accuracy: `0.4700` (47.00%)

Fine-tuning made that model worse, for two reasons. Batch Normalization layers switched to per-batch statistics, so training accuracy dropped from 71% (the restored head-stage epoch 4) to 32% in the first fine-tuning epoch. A shared `EarlyStopping` callback then stopped fine-tuning after 4 epochs without restoring any weights. Both are fixed in the current notebook, and the notebook's analysis section explains them in detail.

## Presentation preparation

The project includes a live presentation. Be prepared to explain:

1. **The problem:** why classifying small, low-resolution RGB images is challenging.
2. **The pipeline:** data, preprocessing, model, training, and evaluation.
3. **The design decisions:** why transfer learning, a frozen stage, fine-tuning, and a reduced learning rate were used.
4. **The results:** test accuracy, loss curves, and the most difficult class pairs.
5. **The limitations:** the 10,000-image training cap, runtime, GPU dependence, and the domain difference between ImageNet and CIFAR-10.
6. **The improvements:** augmentation, a CIFAR-specific architecture, more data, gradual unfreezing, longer training, or per-class metrics.

The presentation should focus on reasoning and insights rather than a line-by-line explanation of the code.

The slides (German, 16 slides with speaker notes) are in [`presentation/`](<./presentation/Fine-Tuning ResNet50 auf CIFAR-10.pptx>).

## Evaluation checklist

The project is evaluated primarily on:

- Correct end-to-end workflow
- Clear and consistent preprocessing
- Intentional use of transfer learning
- Appropriate evaluation and interpretation
- Quality of analytical reasoning
- Clear communication during the presentation

Maximum accuracy and visual polish are less important than demonstrating a reliable workflow and explaining the decisions behind it.

## Project files

| File | Description |
| --- | --- |
| [`cifar10_resnet50_transfer_learning.ipynb`](./notebooks/cifar10_resnet50_transfer_learning.ipynb) | Project notebook (German), executed in Google Colab on a GPU |
| [`Fine-Tuning ResNet50 auf CIFAR-10.pptx`](<./presentation/Fine-Tuning ResNet50 auf CIFAR-10.pptx>) | Presentation slides (German), based on the final run |
| [`README.md`](./README.md) | Project documentation |
| [`results/`](./results/) | `metrics.json`, `training_history.png`, `confusion_matrix.png` of the final run |

## Folder structure

| Folder | Purpose |
| --- | --- |
| [`notebooks/`](./notebooks/) | Executable Jupyter and Google Colab notebook |
| [`presentation/`](./presentation/) | Presentation slides |
| [`results/`](./results/) | Exported metrics and plots |
