#!/usr/bin/env python3
"""
Train BiGRU on WLASL upper-body pose data (184 classes, 7 joints * 3 = 21 features).
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Config
DATA_PATH = "training/data/wlasl_pose.parquet"
WINDOW_SIZE = 16
FEATURE_DIM = 21  # 7 joints * 3 coords
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "public/models/fsl_model/fsl_bigru_wlasl_pose"

def load_wlasl_pose_data(parquet_path, window_size=16, stride=8):
    """Load WLASL pose data with sliding window."""
    df = pd.read_parquet(parquet_path)
    
    # Encode labels
    labels = sorted(df['label'].unique())
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    X, y = [], []
    
    for idx, row in df.iterrows():
        poses = row['poses']  # Object array of (7, 3) arrays
        label = label_to_idx[row['label']]
        
        T = len(poses)
        if T < 16:
            continue
            
        # Stack frames to (T, 21)
        poses_flat = np.stack([np.stack(p).reshape(-1) for p in poses])  # (T, 21)
        
        # Sliding window
        stride = 8
        for i in range(0, len(poses) - 16 + 1, stride):
            window = poses[i:i+16]
            if len(window) == 16:
                window_flat = np.stack([np.stack(w).reshape(-1) for w in window])  # (16, 21)
                X.append(window_flat)
                y.append(label_to_idx[row['label']])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    labels = sorted(set(df['label']))
    print(f"Total windows: {len(X)}")
    print(f"Shape: {X.shape}")
    print(f"Classes: {len(set(df['label']))}")
    
    return X, y, sorted(set(df['label']))

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        batch_size = tf.shape(y_pred)[0]
        indices = tf.stack([tf.range(tf.shape(y_true)[0]), tf.cast(y_true, tf.int32)], axis=1)
        pt = tf.gather_nd(y_pred, indices)
        
        gamma = 2.0
        alpha_t = 0.25
        
        loss = -alpha_t * tf.pow(1 - pt, gamma) * tf.math.log(pt)
        return tf.reduce_mean(loss)
    return loss

def build_bigru_model(num_classes, feature_dim=21, window_size=16):
    inputs = layers.Input(shape=(16, 21), name="input")
    
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(inputs)
    x = layers.Bidirectional(layers.GRU(128, return_sequences=False, dropout=0.3, recurrent_dropout=0.2))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(184, activation='softmax', name='output')(x)
    
    model = models.Model(inputs, outputs, name="fsl_bigru_wlasl_pose")
    return model

def main():
    print("Loading WLASL pose dataset...")
    X, y, labels = load_wlasl_pose_data("training/data/wlasl_pose.parquet")
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"Classes: {len(set(y))}")
    
    # Build model
    model = build_bigru_model(184, 21, 16)
    
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        batch_size = tf.shape(y_pred)[0]
        indices = tf.stack([tf.range(tf.shape(y_true)[0]), tf.cast(y_true, tf.int32)], axis=1)
        pt = tf.gather_nd(y_pred, indices)
        
        gamma = 2.0
        alpha_t = 0.25
        
        loss = -0.25 * tf.pow(1 - pt, 2.0) * tf.math.log(pt)
        return tf.reduce_mean(loss)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=lambda y_true, y_pred: tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(y_true, y_pred),
        metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
    )
    
    model.summary()
    
    os.makedirs("public/models/fsl_model/fsl_bigru_wlasl_pose", exist_ok=True)
    
    callbacks_list = [
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy'),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6, monitor='val_loss'),
        tf.keras.callbacks.ModelCheckpoint(
            "public/models/fsl_model/fsl_bigru_wlasl_pose/best_model.keras",
            save_best_only=True, monitor='val_accuracy', mode='max'
        ),
        tf.keras.callbacks.CSVLogger("public/models/fsl_model/fsl_bigru_wlasl_pose/training_log.csv")
    ]
    
    # Class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
    
    history = model.fit(
        X, y,
        validation_split=0.2,
        batch_size=32,
        epochs=100,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy'),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6, monitor='val_loss'),
            tf.keras.callbacks.ModelCheckpoint(
                "public/models/fsl_model/fsl_bigru_wlasl_pose/best_model.keras",
                save_best_only=True, monitor='val_accuracy', mode='max'
            ),
        ],
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Save
    os.makedirs("public/models/fsl_model/fsl_bigru_wlasl_pose", exist_ok=True)
    model.save("public/models/fsl_model/fsl_bigru_wlasl_pose/final_model.keras")
    
    label_map = {str(i): label for i, label in enumerate(sorted(set(y)))}
    with open("public/models/fsl_model/fsl_bigru_wlasl_pose/label_map.json", 'w') as f:
        json.dump(label_map, f, indent=2)
    
    print(f"Model saved!")
    val_loss, val_acc, val_topk = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation: Loss={val_loss:.4f}, Acc={val_acc:.4f}, Top-5 Acc={val_topk:.4f}")

if __name__ == "__main__":
    import tensorflow as tf
    main()