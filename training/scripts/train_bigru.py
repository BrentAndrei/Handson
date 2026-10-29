import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

# Config
KAGGLE_DATA_DIR = "training/data/processed_all"
WINDOW_SIZE = 16
FEATURE_DIM = 63  # 21 hand landmarks * 3 coords (right hand only)
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "public/models/fsl_model/fsl_bigru_v2_all54"
NUM_CLASSES = 54  # 28 Kaggle + 26 Transactional = 54 classes

def load_kaggle_data(data_dir, window_size=16, max_samples_per_class=None):
    """Load preprocessed Kaggle FSL dataset."""
    X = np.load(os.path.join(data_dir, "X.npy"))
    y = np.load(os.path.join(data_dir, "y.npy"))
    with open(os.path.join(data_dir, "label_map.json")) as f:
        label_map = json.load(f)
    
    print(f"Loaded {len(X)} windows, shape: {X.shape}")
    print(f"Classes: {len(set(y))}")
    
    return X, y, None  # label_encoder not needed

def build_bigru_model(num_classes, feature_dim=63, window_size=16):
    """Build BiGRU model matching current architecture."""
    inputs = layers.Input(shape=(16, 63), name="input")
    
    # BiGRU layers
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(inputs)
    x = layers.Bidirectional(layers.GRU(128, return_sequences=False, dropout=0.3, recurrent_dropout=0.2))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='output')(x)
    
    model = models.Model(inputs, outputs, name="fsl_bigru")
    return model

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

# Config
KAGGLE_DATA_DIR = "training/data/processed_all83"
WINDOW_SIZE = 16
FEATURE_DIM = 63  # 21 hand landmarks * 3 coords (right hand only)
BATCH_SIZE = 64
EPOCHS = 120
PATIENCE = 20
LEARNING_RATE = 1e-3
MIN_LR = 1e-6
MODEL_SAVE_PATH = "public/models/fsl_model/fsl_bigru_v5_81"
NUM_CLASSES = 81  # 28 Kaggle + 26 Trans NPY + 26 Trans Vid + 29 FSL-105 - overlaps = 81
LABEL_SMOOTHING = 0.1
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

def load_kaggle_data(data_dir, window_size=16, max_samples_per_class=None):
    """Load preprocessed Kaggle FSL dataset."""
    X = np.load(os.path.join(data_dir, "X.npy"))
    y = np.load(os.path.join(data_dir, "y.npy"))
    with open(os.path.join(data_dir, "label_map.json")) as f:
        label_map = json.load(f)
    
    print(f"Loaded {len(X)} windows, shape: {X.shape}")
    print(f"Classes: {len(set(y))}")
    
    return X, y, None  # label_encoder not needed

def focal_loss(y_true, y_pred):
    """Focal loss with label smoothing."""
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    
    # Gather predicted probability for true class
    batch_size = tf.shape(y_pred)[0]
    indices = tf.stack([tf.range(tf.shape(y_true)[0]), tf.cast(y_true, tf.int32)], axis=1)
    pt = tf.gather_nd(y_pred, indices)
    
    # Focal loss: -alpha * (1-pt)^gamma * log(pt)
    gamma = 2.0
    alpha = 0.25
    
    # Class weights
    class_weights = tf.constant([class_weight_dict[i] for i in range(NUM_CLASSES)], dtype=tf.float32)
    alpha_t = tf.gather(class_weights, tf.cast(y_true, tf.int32)) * alpha
    
    loss = -alpha_t * tf.pow(1 - pt, gamma) * tf.math.log(pt)
    return tf.reduce_mean(loss)

def label_smoothing_crossentropy(y_true, y_pred):
    """Label smoothing cross entropy."""
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=NUM_CLASSES)
    y_true = y_true * (1 - LABEL_SMOOTHING) + LABEL_SMOOTHING / NUM_CLASSES
    return tf.keras.losses.CategoricalCrossentropy(from_logits=False)(y_true, y_pred)

def combined_loss(y_true, y_pred):
    """Combined focal + label smoothing loss."""
    focal = focal_loss(y_true, y_pred)
    ls = label_smoothing_crossentropy(y_true, y_pred)
    return 0.7 * focal + 0.3 * ls

def build_bigru_model(num_classes, feature_dim=63, window_size=16):
    """Build improved BiGRU model with attention."""
    inputs = layers.Input(shape=(window_size, feature_dim), name="input")
    
    # BiGRU layers
    x = layers.Bidirectional(layers.GRU(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(inputs)
    x = layers.Bidirectional(layers.GRU(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(x)
    
    # Self-attention
    attention = layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)
    
    x = layers.Bidirectional(layers.GRU(128, return_sequences=False, dropout=0.3, recurrent_dropout=0.2))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='output')(x)
    
    model = models.Model(inputs, outputs, name="fsl_bigru_attention")
    return model

def cosine_annealing_with_warmup(epoch, lr):
    """Cosine annealing with warmup."""
    warmup_epochs = 5
    total_epochs = EPOCHS
    
    if epoch < warmup_epochs:
        return float(LEARNING_RATE * (epoch + 1) / warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return float(MIN_LR + 0.5 * (LEARNING_RATE - MIN_LR) * (1 + np.cos(np.pi * progress)))

class WeightedFocalLoss(tf.keras.losses.Loss):
    def __init__(self, class_weights, gamma=2.0, alpha=0.25, label_smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = tf.constant(class_weights, dtype=tf.float32)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
    
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        batch_size = tf.shape(y_pred)[0]
        indices = tf.stack([tf.range(tf.shape(y_true)[0]), tf.cast(y_true, tf.int32)], axis=1)
        pt = tf.gather_nd(y_pred, indices)
        
        gamma = 2.0
        alpha = 0.25
        
        class_weights_tensor = tf.constant([class_weight_dict[i] for i in range(NUM_CLASSES)], dtype=tf.float32)
        alpha_t = tf.gather(self.class_weights, tf.cast(y_true, tf.int32)) * alpha
        
        # Focal loss
        focal = -alpha_t * tf.pow(1 - pt, self.gamma) * tf.math.log(pt)
        focal_loss = tf.reduce_mean(focal)
        
        # Label smoothing
        y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), depth=NUM_CLASSES)
        y_true_smooth = y_true_onehot * (1 - self.label_smoothing) + self.label_smoothing / NUM_CLASSES
        ls = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(y_true_smooth, y_pred)
        
        return 0.7 * focal_loss + 0.3 * ls

def main():
    print("Loading combined dataset (Kaggle + Transactional)...")
    X, y, _ = load_kaggle_data(KAGGLE_DATA_DIR)
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    global class_weight_dict
    class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
    print("Class weights:", class_weight_dict)
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"Classes: {len(set(y))}")
    
    # Build model
    model = build_bigru_model(NUM_CLASSES, FEATURE_DIM, WINDOW_SIZE)
    
    # Loss with class weights, focal, and label smoothing
    loss_fn = WeightedFocalLoss(
        class_weights=[class_weight_dict[i] for i in range(NUM_CLASSES)],
        gamma=2.0,
        alpha=0.25,
        label_smoothing=0.1
    )
    
    # Cosine annealing with warmup
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch, lr: cosine_annealing_with_warmup(epoch, lr),
        verbose=1
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-4),
        loss=WeightedFocalLoss(
            class_weights=[class_weight_dict[i] for i in range(NUM_CLASSES)],
            gamma=2.0,
            alpha=0.25,
            label_smoothing=0.1
        ),
        metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
    )
    
    model.summary()
    
    # Callbacks
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    callbacks_list = [
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor='val_sparse_top_k_categorical_accuracy', mode='max'),
        tf.keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5, min_lr=MIN_LR, monitor='val_loss', verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            f"{MODEL_SAVE_PATH}/best_model.keras",
            save_best_only=True, monitor='val_sparse_top_k_categorical_accuracy', mode='max'
        ),
        tf.keras.callbacks.CSVLogger(f"{MODEL_SAVE_PATH}/training_log.csv"),
        tf.keras.callbacks.LearningRateScheduler(cosine_annealing_with_warmup, verbose=0),
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Save final model and label map
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.save(f"{MODEL_SAVE_PATH}/final_model.keras")
    
    # Load label map from combined dataset
    with open(os.path.join(KAGGLE_DATA_DIR, "label_map.json")) as f:
        label_map = json.load(f)
    
    with open(f"{MODEL_SAVE_PATH}/label_map.json", 'w') as f:
        json.dump(label_map, f, indent=2)
    
    # Save label encoder classes for inference
    labels = [label_map[str(i)] for i in range(NUM_CLASSES)]
    np.save(f"{MODEL_SAVE_PATH}/label_classes.npy", np.array(labels))
    
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # Evaluate
    val_loss, val_acc, val_topk = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation: Loss={val_loss:.4f}, Acc={val_acc:.4f}, Top-5 Acc={val_topk:.4f}")

if __name__ == "__main__":
    import tensorflow as tf
    main()