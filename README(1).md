# Facial Emotion Recognition (FER2013 + CNN)

A lightweight, Keras/TensorFlow-based pipeline for classifying facial
expressions from grayscale face images using the FER2013 dataset.

## Overview

This project provides:

- **`load_and_process.py`** — loads the FER2013 CSV dataset, decodes the
  pixel strings into images, resizes them, and prepares train-ready
  arrays with one-hot encoded emotion labels.
- **`cnn_models_improved.py`** — several CNN architectures
  (`simple_CNN`, `simpler_CNN`, `tiny_XCEPTION`, `mini_XCEPTION`,
  `big_XCEPTION`) for classifying the processed face images into
  emotion categories, using batch normalization, L2 regularization,
  and global average pooling to reduce overfitting.

## Dataset

This project uses the [FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
dataset, which contains 48x48 grayscale face images labeled with one of
seven emotions:

| Label | Emotion  |
|-------|----------|
| 0     | Angry    |
| 1     | Disgust  |
| 2     | Fear     |
| 3     | Happy    |
| 4     | Sad      |
| 5     | Surprise |
| 6     | Neutral  |

Download `fer2013.csv` and place it in the project root (or update
`dataset_path` in `load_and_process.py`).

## Project Structure

```
.
├── load_and_process.py       # dataset loading and preprocessing
├── cnn_models_improved.py    # model architectures
├── fer2013.csv                # dataset (not included, download separately)
└── README.md
```

## Requirements

```
tensorflow>=2.10
pandas
numpy
opencv-python
```

Install with:

```bash
pip install tensorflow pandas numpy opencv-python
```

## Usage

### 1. Load and preprocess the data

```python
from load_and_process import load_fer2013, preprocess_input

faces, emotions = load_fer2013()
faces = preprocess_input(faces)
```

### 2. Split into train/validation/test sets

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    faces, emotions, test_size=0.2, random_state=42
)
```

### 3. Build and train a model

```python
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cnn_models_improved import mini_XCEPTION

input_shape = (48, 48, 1)
num_classes = 7

model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True),
]

model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    validation_data=(x_test, y_test),
    epochs=100,
    callbacks=callbacks
)
```

## Model Options

| Model            | Params (approx.) | Notes                                    |
|------------------|-------------------|-------------------------------------------|
| `simple_CNN`     | Larger            | Straightforward stacked conv layers       |
| `simpler_CNN`    | Medium            | Strided convs instead of pooling          |
| `tiny_XCEPTION`  | Very small         | Depthwise separable convs, residual blocks |
| `mini_XCEPTION`  | Small             | Good accuracy/size tradeoff (recommended) |
| `big_XCEPTION`   | Largest            | Best accuracy, slower to train            |

`mini_XCEPTION` is generally the best starting point: it trains fast,
generalizes well on FER2013, and is small enough to run in real time
(e.g., on video/webcam input).

## Notes

- `preprocess_input(x, v2=True)` scales pixel values to the `[-1, 1]`
  range (recommended for the XCEPTION-style models). Set `v2=False`
  to instead scale to `[0, 1]`.
- The original `load_fer2013()` function used `data['emotion'].as_matrix()`,
  which was removed in modern pandas. Use `.to_numpy()` or
  `.values` with recent pandas versions.

## License

Add your preferred license here (e.g., MIT).
