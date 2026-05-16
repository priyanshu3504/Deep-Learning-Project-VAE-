import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

# LOAD DATA

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encode labels
num_classes = 10

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

latent_dim = 128

# ENCODER

image_inputs = tf.keras.Input(shape=(32, 32, 3))
label_inputs = tf.keras.Input(shape=(num_classes,))

# Expand label dimensions
label_embedding = layers.Dense(32 * 32)(label_inputs)
label_embedding = layers.Reshape((32, 32, 1))(label_embedding)

# Concatenate image and label
x = layers.Concatenate()([image_inputs, label_embedding])

x = layers.Conv2D(
    32,
    3,
    strides=2,
    activation="relu",
    padding="same"
)(x)

x = layers.Conv2D(
    64,
    3,
    strides=2,
    activation="relu",
    padding="same"
)(x)

x = layers.Flatten()(x)

x = layers.Dense(256, activation="relu")(x)

# Mean and variance
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

# REPARAMETERIZATION

def sampling(args):
    z_mean, z_log_var = args

    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))

    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

encoder = tf.keras.Model(
    [image_inputs, label_inputs],
    [z_mean, z_log_var, z],
    name="encoder"
)

# DECODER

latent_inputs = tf.keras.Input(shape=(latent_dim,))
label_inputs_decoder = tf.keras.Input(shape=(num_classes,))

# Combine latent vector and label
x = layers.Concatenate()([latent_inputs, label_inputs_decoder])

x = layers.Dense(8 * 8 * 64, activation="relu")(x)

x = layers.Reshape((8, 8, 64))(x)

x = layers.Conv2DTranspose(
    64,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(x)

x = layers.Conv2DTranspose(
    32,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(x)

decoder_outputs = layers.Conv2DTranspose(
    3,
    3,
    activation="sigmoid",
    padding="same"
)(x)

decoder = tf.keras.Model(
    [latent_inputs, label_inputs_decoder],
    decoder_outputs,
    name="decoder"
)

# CVAE MODEL

class CVAE(tf.keras.Model):

    def __init__(self, encoder, decoder, **kwargs):
        super(CVAE, self).__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):

        images, labels = data

        with tf.GradientTape() as tape:

            z_mean, z_log_var, z = self.encoder([images, labels])

            reconstruction = self.decoder([z, labels])

            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(
                        images,
                        reconstruction
                    ),
                    axis=(1, 2)
                )
            )

            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var
                    - tf.square(z_mean)
                    - tf.exp(z_log_var),
                    axis=1
                )
            )

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(
            zip(grads, self.trainable_weights)
        )

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss
        }

# TRAIN MODEL

cvae = CVAE(encoder, decoder)

cvae.compile(optimizer=tf.keras.optimizers.Adam())

cvae.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=128
)

# GENERATE CLASS-SPECIFIC IMAGES

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# Choose class
selected_class = 5  # dog

label = tf.keras.utils.to_categorical(
    [selected_class],
    num_classes
)

random_latent_vectors = tf.random.normal(
    shape=(16, latent_dim)
)

labels = np.repeat(label, 16, axis=0)

generated_images = decoder.predict(
    [random_latent_vectors, labels],
    verbose=0
)

# DISPLAY IMAGES

plt.figure(figsize=(8, 8))

for i in range(16):

    ax = plt.subplot(4, 4, i + 1)

    plt.imshow(generated_images[i])

    plt.axis("off")

plt.suptitle(
    f"Generated Class: {class_names[selected_class]}"
)

plt.tight_layout()

plt.show()