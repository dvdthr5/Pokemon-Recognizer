import os, random, numpy as np

SEED = int(os.getenv("SEED", "1337"))
MODEL_PATH = os.getenv("MODEL_PATH", "pokemon_model.keras")
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", "class_names.json")
TEST_FOLDER = os.getenv("TEST_FOLDER", "test_images")

def set_seed():
    import tensorflow as tf
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)