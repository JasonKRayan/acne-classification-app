# PreProcessing
Dont commit the raw dataset, add to .gitignore

## Model and Image Processing

Model: EfficientNet-Lite4
Load & decode (RGB, 3 channels)

Convert to float32 and scale to [0, 1]

Resize shortest side → 256 px (preserve aspect ratio)

Center crop → 224×224

Return (image, label_idx)

## Split

Train: ~85% of original train

Validation: ~15%

Test: from dataset’s provided test folder

## Using preprocessing.py

import pandas as pd
from preprocessing import df_to_dataset

train_df = pd.read_csv("processed_splits/processed_train.csv")
val_df   = pd.read_csv("processed_splits/processed_val.csv")

train_ds = df_to_dataset(train_df, shuffle=True)
val_ds   = df_to_dataset(val_df, shuffle=False)

for images, labels in train_ds.take(1):
    print(images.shape)   # (batch, 224, 224, 3)
    print(labels.shape)




