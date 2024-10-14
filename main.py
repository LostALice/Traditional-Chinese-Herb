# Code by AkinoAlice@Tyrant_Rex

from PIL import Image

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os


class Classifier(object):
    def __init__(self) -> None:
        ...

    def predict_img(
        self,
        model_path: str = "./models/model.h5",
        target_path: str = "./test/image.jpg"
    ) -> tuple[np.ndarray, str]:

        model = tf.keras.models.load_model(model_path)
        self.class_names = ["Arecacatechu",
                            "gastrodiaelata",
                            "Ligusticum",
                            "Liriope",
                            "nuxvomica",
                            "Pinelliaternate",
                            "radixcurcumae",
                            "None"]

        img = Image.open(target_path).resize((224, 224))
        img_array = np.array(img) / 255.0

        if img_array.shape[-1] != 3:
            img_array = np.stack([img_array] * 3, axis=-1)

        img_array = np.expand_dims(img_array, axis=0)

        outputs = model.predict(img_array)
        # outputs = outputs if outputs else -1

        result_index = np.argmax(outputs)
        result = self.class_names[result_index]

        return outputs, result


class Dataset(object):
    def __init__(self, dataset_path: str = "./image") -> None:
        folders = os.listdir(dataset_path)
        self.names = []
        self.nums = []

        for folder in folders:
            folder_path = os.path.join(dataset_path, folder)
            images = os.listdir(folder_path)
            images_num = len(images)
            print(f"{folder}:{images_num}")
            self.names.append(folder)
            self.nums.append(images_num)

        self.data = (self.names, self.nums)

    def show_graph(self) -> None:
        x, y = self.data
        plt.xlabel("num")
        plt.plot(x, y)
        plt.title("Num of medicinal materials")
        plt.show()


class Trainer(object):
    def __init__(self) -> None:
        ...

    def load_data(
        self,
        data_dir: str = "./image",
        img_height: int = 224,
        img_width: int = 224,
        batch_size: int = 4
    ) -> tuple:

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            label_mode="categorical",
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            label_mode="categorical",
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        class_names = train_ds.class_names

        return train_ds, val_ds, class_names

    def load_model(
        self,
        image_shape: tuple[int, int, int] = (224, 224, 3)
    ) -> tf.keras.Model:

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=image_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(7, activation="softmax")
        ])

        model.summary()
        model.compile(optimizer="adam",
                      loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    def train(
        self,
        epochs: int = 10,
        model_path: str = "./model/model.h5",
        show_history: bool = False,
        data_dir: str = "./image",
        img_height: int = 224,
        img_width: int = 224,
        batch_size: int = 4
    ) -> None:

        train_ds, val_ds, class_names = self.load_data(
            data_dir=data_dir,
            img_height=img_height,
            img_width=img_width,
            batch_size=batch_size
        )

        self.model = self.load_model()
        self.history = self.model.fit(
            train_ds, validation_data=val_ds, epochs=epochs)

        if show_history:
            self.show_loss_acc()

        self.model.save(model_path)

    def show_loss_acc(self) -> None:
        acc = self.history.history["accuracy"]
        val_acc = self.history.history["val_accuracy"]

        loss = self.history.history["loss"]
        val_loss = self.history.history["val_loss"]

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label="Training Accuracy")
        plt.plot(val_acc, label="Validation Accuracy")
        plt.legend(loc="lower right")
        plt.ylabel("Accuracy")
        plt.ylim([min(plt.ylim()), 1])
        plt.title("Training and Validation Accuracy")

        plt.subplot(2, 1, 2)
        plt.plot(loss, label="Training Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.legend(loc="upper right")
        plt.ylabel("Cross Entropy")
        plt.ylim([0, 1.0])
        plt.title("Training and Validation Loss")
        plt.xlabel("epoch")
        plt.show()

    def load_test_data(
        self, test_data_path: str = "./test",
        img_height: int = 224,
        img_width: int = 224,
        batch_size: int = 4
    ) -> tuple:

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            test_data_path,
            label_mode="categorical",
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            test_data_path,
            label_mode="categorical",
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        class_names = train_ds.class_names

        return train_ds, val_ds, class_names

    def test(self, model_path: str = "./models/model.h5") -> None:
        train_ds, val_ds, class_names = self.load_test_data(
            "./Images/", 224, 224, 4)
        model = tf.keras.models.load_model(model_path)
        model.summary()
        loss, accuracy = model.evaluate(val_ds)
        print("Test accuracy :", accuracy, loss)

if __name__ == "__main__":
    # load dataset
    data = Dataset(dataset_path="./image")

    # train
    Trainer().train(epochs=10, show_history=True)

    # classify
    predict, result = Classifier().predict_img(model_path="./model/model.h5", target_path="./test/image.jpg")
    print(result)