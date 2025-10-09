import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2

# --- CONFIG ---
tf.config.run_functions_eagerly(True)  # ensure eager execution for safety/debug
DATA_DIR = "data"
MODEL_PATH = "pokemon_model.keras"      # use modern Keras format
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
EPOCHS = 10

print("📂 Loading dataset...")

# --- LOAD TRAIN AND TEST DATASETS ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# --- SAVE CLASS NAMES ---
class_names = train_ds.class_names
num_classes = len(class_names)

with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

print(f"💾 Saved {num_classes} class names to {CLASS_NAMES_PATH}")
print(f"🧩 Found {num_classes} Pokémon classes.")

# --- DATA AUGMENTATION PIPELINE ---
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# --- APPLY AUGMENTATION TO TRAIN DATA ---
train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# --- LOAD OR BUILD MODEL ---
if os.path.exists(MODEL_PATH):
    print(f"🔁 Loading existing model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH, compile=False)
else:
    print("🆕 Creating new model with MobileNetV2 backbone...")

    base_model = MobileNetV2(input_shape=IMG_SIZE + (3,),
                             include_top=False,
                             weights="imagenet")
    base_model.trainable = False  # freeze pretrained convolutional layers

    # Add custom classifier head
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=base_model.input, outputs=output)

# --- CHECK CLASS MISMATCH ---
if model.output_shape[-1] != num_classes:
    print("⚠️ Model output size differs — rebuilding classification head...")

    base_model = None
    if "mobilenetv2" in model.name.lower():
        base_model = model.layers[0]  # reuse pretrained backbone if available
    else:
        base_model = MobileNetV2(input_shape=IMG_SIZE + (3,),
                                 include_top=False,
                                 weights="imagenet")
        base_model.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=base_model.input, outputs=output)

# --- COMPILE MODEL ---
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# --- SETUP CHECKPOINTING AND LOGGING ---
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
    verbose=1
)
tensorboard_cb = keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1)

# --- TRAIN MODEL ---
print("🚀 Training (fine-tuning) model...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, tensorboard_cb]
)

# --- OPTIONAL: UNFREEZE TOP LAYERS FOR FINE-TUNING ---
print("🔧 Unfreezing top MobileNetV2 layers for fine-tuning...")
base_model = model.layers[0] if isinstance(model.layers[0], keras.Model) else None
if base_model:
    for layer in base_model.layers[:-40]:  # keep earlier layers frozen
        layer.trainable = False
    for layer in base_model.layers[-40:]:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    print("🚀 Continuing fine-tuning for a few more epochs...")
    fine_tune_epochs = 5
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_tune_epochs,
        callbacks=[checkpoint_cb, tensorboard_cb]
    )

# --- SAVE FINAL MODEL ---
model.save(MODEL_PATH)
print(f"✅ Final model saved to {MODEL_PATH}")

# --- PRINT TRAINING RESULTS ---
final_acc = history.history["accuracy"][-1]
final_val_acc = history.history["val_accuracy"][-1]
print(f"📈 Final Training Accuracy: {final_acc:.3f}, Validation Accuracy: {final_val_acc:.3f}")
print("\n💡 Tip: Run 'tensorboard --logdir logs' to visualize training progress in your browser.")
