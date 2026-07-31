# ================================
# ROAD SEGMENTATION USING FCN (FINAL CORRECT)
# ================================

!pip install opencv-python matplotlib scikit-learn

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# -------------------------------
# FIXED PATH (YOUR DATASET)
# -------------------------------
BASE_PATH = "dataset/CamVid"

train_p = BASE_PATH + "/train"
train_l = BASE_PATH + "/train_labels"
val_p = BASE_PATH + "/val"
val_l = BASE_PATH + "/val_labels"
test_p = BASE_PATH + "/test"
test_l = BASE_PATH + "/test_labels"

# -------------------------------
# LOAD DATA (FINAL FIXED)
# -------------------------------
IMG_SIZE = 224

def load_data(img_path, mask_path):
    images, masks = [], []

    img_files = sorted(os.listdir(img_path))

    for img_name in img_files:
        img_file = os.path.join(img_path, img_name)
        base = os.path.splitext(img_name)[0]

        # Match mask file
        possible_masks = [
            base + ".png",
            base + ".jpg",
            base + "_L.png",
            base + "_label.png"
        ]

        mask_file = None
        for m in possible_masks:
            path = os.path.join(mask_path, m)
            if os.path.exists(path):
                mask_file = path
                break

        img = cv2.imread(img_file)

        if img is None or mask_file is None:
            continue

        mask = cv2.imread(mask_file)

        if mask is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Road class (CamVid color)
        road = np.all(mask == [128, 64, 128], axis=-1)

        mask = road.astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)  # 🔥 IMPORTANT FIX

        images.append(img)
        masks.append(mask)

    print("Loaded:", len(images), "images from", img_path)

    return np.array(images), np.array(masks)

# Load datasets
X_train, Y_train = load_data(train_p, train_l)
X_val, Y_val = load_data(val_p, val_l)

print("Train:", X_train.shape, Y_train.shape)
print("Val:", X_val.shape, Y_val.shape)

# -------------------------------
# FCN MODEL
# -------------------------------
def build_fcn():
    inputs = layers.Input(shape=(224,224,3))

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)

    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)

    return models.Model(inputs, outputs)

model = build_fcn()
model.summary()

# -------------------------------
# IoU METRIC
# -------------------------------
def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-7)

# -------------------------------
# COMPILE
# -------------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', iou_metric]
)

# -------------------------------
# TRAIN
# -------------------------------
model.fit(
    X_train, Y_train,
    epochs=50,
    batch_size=8,
    validation_data=(X_val, Y_val)
)

# -------------------------------
# TEST
# -------------------------------
X_test, Y_test = load_data(test_p, test_l)

pred = model.predict(X_test[0:1])[0]
pred_mask = (pred > 0.5).astype(np.uint8)

# -------------------------------
# SHOW RESULT
# -------------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(X_test[0])
plt.title("Input")

plt.subplot(1,3,2)
plt.imshow(Y_test[0].squeeze())
plt.title("Actual")

plt.subplot(1,3,3)
plt.imshow(pred_mask.squeeze())
plt.title("Predicted")

plt.show()

# -------------------------------
# SAVE MODEL
# -------------------------------
model.save("road_segmentation_fcn.h5")
