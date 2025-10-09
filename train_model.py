import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

# --- PARAMETERS ---
img_size = (128, 128)
batch_size = 32
train_dir = "data/train"
test_dir = "data/test"

# --- LOAD DATA ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical"
)

# ✅ Capture class names BEFORE mapping or prefetching
class_names = train_ds.class_names
print("Classes found:", class_names)

# --- NORMALIZE PIXELS ---
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
test_ds = test_ds.map(lambda x, y: (x / 255.0, y))

# --- CACHE AND PREFETCH FOR PERFORMANCE ---
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# --- DEFINE MODEL ---
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=img_size + (3,)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])

# --- COMPILE MODEL ---
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- SUMMARY ---
model.summary()

# --- TRAIN MODEL ---
epochs = 10
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=epochs
)

# --- SAVE MODEL ---
os.makedirs("model", exist_ok=True)
model.save("model/pokemon_classifier.h5")
print("\n✅ Model training complete! Saved as model/pokemon_classifier.h5")

# --- PLOT TRAINING RESULTS ---
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Pokémon Classifier Training Progress")
plt.legend()
plt.show()
