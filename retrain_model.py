import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- CONFIG ---
tf.config.run_functions_eagerly(False)  # ensure graph execution for better optimization
DATA_DIR = "data"
MODEL_PATH = "pokemon_model.keras"      # use modern Keras format
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
EPOCHS = 20

# --- SET MIXED PRECISION POLICY IF SUPPORTED ---
try:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_bfloat16')
    mixed_precision.set_policy(policy)
    print(f"⚡ Mixed precision policy set to: {policy}")
except Exception:
    # fallback if mixed precision is not supported
    print("⚡ Mixed precision not supported, using default float32 policy")

print("📂 Loading dataset...")

# --- LOAD TRAIN AND TEST DATASETS ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)


# --- SAVE CLASS NAMES ---
class_names = train_ds.class_names
num_classes = len(class_names)

with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

print(f"💾 Saved {num_classes} class names to {CLASS_NAMES_PATH}")
print(f"🧩 Found {num_classes} Pokémon classes.")

# --- DATA PREPROCESSING AND AUGMENTATION PIPELINE ---
preprocess_layer = layers.Lambda(preprocess_input)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
])

def preprocess_and_augment(x, y):
    x = preprocess_layer(x)
    x = data_augmentation(x)
    return x, y

# --- APPLY PREPROCESSING, SHUFFLING, AND AUGMENTATION TO TRAIN DATA ---
train_ds = train_ds.shuffle(1000)
train_ds = train_ds.map(preprocess_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

print("📦 Loading previously trained model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully, continuing training...")
optimizer = Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer,
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# --- SETUP CALLBACKS ---
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
    verbose=1
)
early_stopping_cb = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
    verbose=1
)
reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    verbose=1,
    min_lr=1e-6
)
tensorboard_cb = keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1)

# --- TRAIN CLASSIFIER HEAD ---
print("🚀 Training classifier head with frozen base model...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb, tensorboard_cb],
    verbose=1
)

# --- UNFREEZE TOP LAYERS FOR FINE-TUNING ---
print("🔧 Unfreezing top MobileNetV2 layers for fine-tuning...")
base_model.trainable = True
for layer in base_model.layers[:-40]:  # keep earlier layers frozen
    layer.trainable = False
for layer in base_model.layers[-40:]:
    layer.trainable = True
# Freeze BatchNormalization layers during fine-tuning
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

print("🚀 Continuing fine-tuning for a few more epochs...")
fine_tune_epochs = 5
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=fine_tune_epochs,
    callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb, tensorboard_cb],
    verbose=1
)

# --- SAVE FINAL MODEL ---
model.save(MODEL_PATH)
print(f"✅ Final model saved to {MODEL_PATH}")

# --- PRINT TRAINING RESULTS ---
final_acc = history.history["accuracy"][-1]
final_val_acc = history.history["val_accuracy"][-1]
print(f"📈 Final Training Accuracy: {final_acc:.3f}, Validation Accuracy: {final_val_acc:.3f}")
print("\n💡 Tip: Run 'tensorboard --logdir logs' to visualize training progress in your browser.")
