import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

decoder = tf.keras.models.load_model("models/decoder.keras")

latent_dim = 128

z1 = np.random.normal(size=(1, latent_dim))
z2 = np.random.normal(size=(1, latent_dim))

n = 10

fig, axes = plt.subplots(1, n, figsize=(20, 4))

for i, alpha in enumerate(np.linspace(0, 1, n)):

    z = (1 - alpha) * z1 + alpha * z2

    generated = decoder.predict(z)

    axes[i].imshow(generated[0])
    axes[i].axis("off")

plt.show()