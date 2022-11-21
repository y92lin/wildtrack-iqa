from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np
import os


def create_encoder(latent_dim=100):
    encoder_inputs = keras.Input(shape=(512, 512, 1))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")
    return encoder


def create_decoder(latent_dim=100):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(128 * 128 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((128, 128, 64))(x)
    x = layers.Conv2DTranspose(64, 1, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 1, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2D(1, 1, activation="relu", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder


class Sampler(layers.Layer):
    def call(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, z_size))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = Sampler()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.mse = keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.SUM)
        self.layer = layers.Normalization(axis=None)
        self.layer.adapt(np.arange(0, 256, dtype=int))

    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker]

    def train_step(self, data):
        x, y = data
        x = self.layer(x)
        y = self.layer(y)

        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(x)
            z = self.sampler(z_mean, z_log_var)
            reconstruction = self.decoder(z)
            reconstruction_loss = self.mse(y, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        # Unpack the data
        x, y = data
        x = self.layer(x)
        y = self.layer(y)
        # Compute predictions
        z_mean, z_log_var = self.encoder(x)
        z = self.sampler(z_mean, z_log_var)
        reconstruction = self.decoder(z)
        reconstruction_loss = self.mse(y, reconstruction)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)
        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def save_model(self, dir_path):
        encoder_path = dir_path + '/encoder'
        decoder_path = dir_path + '/decoder'

        if not os.path.exists(encoder_path):
            os.makedirs(encoder_path)
        if not os.path.exists(decoder_path):
            os.makedirs(decoder_path)

        self.encoder.save_weights(encoder_path, overwrite=True)
        self.decoder.save_weights(decoder_path, overwrite=True)

    def load_model(self, dir_path):
        encoder_path = dir_path + '/encoder'
        decoder_path = dir_path + '/decoder'

        if not os.path.exists(encoder_path):
            os.makedirs(encoder_path)
        if not os.path.exists(decoder_path):
            os.makedirs(decoder_path)

        self.encoder.load_weights(encoder_path)
        self.decoder.load_weights(decoder_path)
