import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from tensorflow.keras.datasets import cifar10


# Custom sampling function
def sampling(args):
    z_mean, z_log_var = args

    epsilon = tf.random.normal(shape=tf.shape(z_mean))

    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Load encoder
encoder = tf.keras.models.load_model(
    "models/encoder.keras",
    custom_objects={"sampling": sampling}
)

# Load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize
x_test = x_test.astype("float32") / 255.0

# Get latent vectors
z_mean, _, _ = encoder.predict(x_test)

# t-SNE
tsne = TSNE(n_components=2)

z_2d = tsne.fit_transform(z_mean)

# Plot
plt.figure(figsize=(10, 10))

plt.scatter(
    z_2d[:, 0],
    z_2d[:, 1],
    c=y_test.flatten(),
    cmap="tab10"
)

plt.colorbar()

plt.title("Latent Space Visualization")

plt.show()