import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np


# LOAD CIFAR-10 DATASET

(x_train, _), (x_test, _) = cifar10.load_data()

# Normalize images between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# LATENT SPACE DIMENSION

latent_dim = 128


# ENCODER

encoder_inputs = tf.keras.Input(shape=(32, 32, 3))

x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)

x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)

# Mean and log variance
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)


# REPARAMETERIZATION TRICK

def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

encoder.summary()


# DECODER

latent_inputs = tf.keras.Input(shape=(latent_dim,))

x = layers.Dense(8 * 8 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((8, 8, 64))(x)

x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)

decoder_outputs = layers.Conv2DTranspose(
    3,
    3,
    activation="sigmoid",
    padding="same"
)(x)

decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")

decoder.summary()


# VAE MODEL

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder


    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed

    def train_step(self, data):

        with tf.GradientTape() as tape:

            z_mean, z_log_var, z = self.encoder(data)

            reconstruction = self.decoder(z)

            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)
                )
            )

            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1
                )
            )

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


# TRAIN MODEL

vae = VAE(encoder, decoder)

vae.compile(optimizer=tf.keras.optimizers.Adam(),loss='mse')

vae.fit(
    x_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_test, x_test)
)

# GENERATE NEW IMAGES


random_latent_vectors = tf.random.normal(shape=(16, latent_dim))

generated_images = decoder(random_latent_vectors)

generated_images = generated_images.numpy()


# DISPLAY GENERATED IMAGES

plt.figure(figsize=(8, 8))

for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(generated_images[i])
    plt.axis("off")

plt.tight_layout()

plt.savefig("outputs/generated_images.png")

plt.show()

encoder.save("models/encoder.keras")
decoder.save("models/decoder.keras")