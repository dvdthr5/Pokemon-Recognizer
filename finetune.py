import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

# ==============================
# CONFIGURATION
# ==============================
TRAIN_DIR = "data/train"   # <-- change this to your train folder path
TEST_DIR = "data/test"     # <-- change this to your test folder path
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 1215
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10

print("[INFO] Building EfficientNetB0 base model...")
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights='imagenet',          # pretrained RGB weights
    input_shape=(224, 224, 3)    # force 3 channels
)
base_model.trainable = False
print("[INFO] Base model built.")

# ==============================
# LOAD DATASET
# ==============================
print("[INFO] Loading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    color_mode='rgb'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    color_mode='rgb'
)

# Normalize pixel values
def normalize_img(image, label):
    return image / 255.0, label

train_ds = train_ds.map(normalize_img)
val_ds = val_ds.map(normalize_img)
# Convert grayscale to RGB if necessary
train_ds = train_ds.map(lambda x, y: (tf.image.grayscale_to_rgb(x) if x.shape[-1] == 1 else x, y))
val_ds = val_ds.map(lambda x, y: (tf.image.grayscale_to_rgb(x) if x.shape[-1] == 1 else x, y))

# Shuffle and prefetch for performance
train_ds = train_ds.shuffle(10000).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# ==============================
# DATA AUGMENTATION
# ==============================
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])


# ==============================
# BUILD MODEL
# ==============================
inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = keras.Model(inputs, outputs)

# ==============================
# COMPILE & INITIAL TRAINING
# ==============================
print("[INFO] Training top layers...")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS
)

# ==============================
# FINE-TUNING
# ==============================
print("[INFO] Fine-tuning upper layers...")

base_model.trainable = True
# Optionally keep earlier layers frozen (EfficientNet has ~237 layers)
for layer in base_model.layers[:200]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=INITIAL_EPOCHS
)

# ==============================
# SAVE MODEL
# ==============================
print("[INFO] Saving model...")
model.save("pokemon_classifier_finetuned.h5")

print("[INFO] Training complete! Model saved as pokemon_classifier_finetuned.h5")
