import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

IMG_SIZE = 128


def create_dataset(samples=300):
    images = []
    masks = []

    for _ in range(samples):
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        mask = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)

        x = np.random.randint(30, 100)
        y = np.random.randint(30, 100)
        r = np.random.randint(10, 25)

        cv2.circle(img, (x, y), r, (255, 255, 255), -1)
        cv2.circle(mask, (x, y), r, 255, -1)

        images.append(img / 255.0)
        masks.append(mask / 255.0)

    return np.array(images), np.array(masks)


def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", activation="relu"
    )(x)
    return x


def build_unet():
    inputs = tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 3))

    # Encoder
    c1 = conv_block(inputs, 32)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128)
    p3 = tf.keras.layers.MaxPooling2D()(c3)

    # Bottleneck
    bn = conv_block(p3, 256)

    # Decoder
    u1 = tf.keras.layers.UpSampling2D()(bn)
    concat1 = tf.keras.layers.Concatenate()([u1, c3])
    c4 = conv_block(concat1, 128)

    u2 = tf.keras.layers.UpSampling2D()(c4)
    concat2 = tf.keras.layers.Concatenate()([u2, c2])
    c5 = conv_block(concat2, 64)

    u3 = tf.keras.layers.UpSampling2D()(c5)
    concat3 = tf.keras.layers.Concatenate()([u3, c1])
    c6 = conv_block(concat3, 32)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(c6)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    x, y = create_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=1
    )

    model = build_unet()
    model.summary()

    model.fit(
        x_train,
        y_train,
        validation_split=0.3,
        epochs=5,
        batch_size=8,
    )

    loss, acc = model.evaluate(x_test, y_test)
    print("Loss:", loss)
    print("Accuracy:", acc)

    preds = model.predict(x_test[:3])

    for i in range(3):
        plt.figure(figsize=(10, 3))

        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(x_test[i])
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Mask")
        plt.imshow(y_test[i].squeeze(), cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Prediction Mask")
        plt.imshow(preds[i].squeeze(), cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
