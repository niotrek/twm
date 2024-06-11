import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras_cv.layers import RandomShear
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from tensorflow.keras.losses import CategoricalCrossentropy

class LetterClassifier:
    def __init__(self, data_dir=None, img_height=28, img_width=28, batch_size=32):
        self.data_dir = data_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.train_ds = None
        self.val_ds = None
        self.model = None
        self.history = None
        self.class_names = []
        self.data_augmentation = self.data_augmentation_layer()
        self.best_model_path = None
        self.load_data()

    def load_data(self):
        self.train_ds = image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            label_mode='categorical',
            verbose=False,
            subset="training",
            seed=123,
            color_mode='grayscale',
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )
        
        self.val_ds = image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            label_mode='categorical',
            verbose=False,
            subset="validation",
            seed=123,
            color_mode='grayscale',
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )
        
        self.class_names = self.train_ds.class_names
        #print(self.class_names)

    def show_sample_images(self):
        plt.figure(figsize=(10, 10))
        for images, labels in self.train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"), cmap='gray')
                plt.title(self.class_names[labels[i]])
                plt.axis("off")
        plt.show()

    def data_augmentation_layer(self):
        data_augmentation = tf.keras.Sequential([
            layers.RandomRotation(0.03, input_shape=(self.img_height, self.img_width, 1)),
            RandomShear((0.2, 0.2)),
            layers.Resizing(self.img_height, self.img_width),
            layers.RandomZoom(0.2),
        ])
        return data_augmentation

    def show_augmented_images(self):
        plt.figure(figsize=(10, 10))
        for images, _ in self.train_ds.take(1):
            for i in range(9):
                augmented_images = self.data_augmentation(images)
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(augmented_images[0].numpy().astype("uint8"), cmap='gray')
                plt.axis("off")
        plt.subplot(3, 3, 1)
        plt.imshow(images[0].numpy().astype("uint8"), cmap='gray')
        plt.axis("off")
        plt.show()

    def build_model(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss_function = CategoricalCrossentropy()

        self.model = models.Sequential()
        self.model.add(self.data_augmentation)
        self.model.add(layers.Rescaling(1./255))
        self.model.add(layers.Conv2D(
            filters = 32,
            kernel_size = (3, 3),
            activation = 'relu',
            kernel_initializer = 'he_uniform',
            padding = 'same'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Conv2D(
            filters = 32,
            kernel_size = (3, 3),
            activation = 'relu',
            kernel_initializer = 'he_uniform',
            padding = 'same'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling2D(
            pool_size = (2, 2)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Conv2D(
            filters = 64,
            kernel_size = (3, 3),
            activation = 'relu',
            kernel_initializer = 'he_uniform',
            padding = 'same'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Conv2D(
            filters = 64,
            kernel_size = (3, 3),
            activation = 'relu',
            kernel_initializer = 'he_uniform',
            padding = 'same'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling2D(
            pool_size = (2, 2)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Conv2D(
            filters = 128,
            kernel_size = (3, 3),
            activation = 'relu',
            kernel_initializer = 'he_uniform',
            padding = 'same'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Conv2D(
            filters = 128,
            kernel_size = (3, 3),
            activation = 'relu',
            kernel_initializer = 'he_uniform',
            padding = 'same'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling2D(
            pool_size = (2, 2)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(
            units = 128,
            activation = 'relu',
            kernel_initializer = 'he_uniform'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(
            rate = 0.4))
        self.model.add(layers.Dense(
            units = 256,
            activation = 'relu',
            kernel_initializer = 'he_uniform'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(
            rate = 0.4))
        self.model.add(layers.Dense(
            units = len(self.class_names),
            activation = 'softmax'))



        self.model.compile(optimizer=optimizer,
                           loss=loss_function,
                           metrics=['accuracy'])
        
        self.model.summary()

    def train_model(self, epochs=300):
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.best_model_path = f"models/best_model_{now}.keras"
        
        early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(self.best_model_path, monitor="val_accuracy", save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5)

        callbacks = [early_stopping, model_checkpoint, reduce_lr]

        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=callbacks
        )

    def load_best_model(self, path=None):
        if path == None:
            path=self.best_model_path
        if os.path.exists(path):
            self.model = models.load_model(path)
            print(f"Loaded best model from {path}")
        else:
            print("Best model path is not set or model file does not exist")

    def evaluate_model(self):
        if self.model is None:
            raise ValueError("Model is not built or loaded.")
        
        loss, accuracy = self.model.evaluate(self.val_ds)
        print(f"Accuracy: {accuracy*100:.2f}%")
        
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def predict_image(self, image):
        if self.model is None:
            raise ValueError("Model is not built or loaded.")
        
        img_array = tf.expand_dims(image, 0)

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(self.class_names[np.argmax(score)], 100 * np.max(score))
        )
        plt.imshow(image, cmap='gray')
        plt.title("Loaded Image")
        plt.axis("off")
        plt.show()
    
    def predict_folder(self, folder_path):
        if os.path.exists(folder_path):
            # Pobranie listy plików w folderze
            image_files = [file for file in os.listdir(folder_path) if file.endswith('.png')]
            for image_file in image_files:
                # Tworzenie pełnej ścieżki do pliku
                image_path = os.path.join(folder_path, image_file)
                
                # Wywołanie funkcji process_image dla każdego obrazu
                self.predict_image(image_path)
            else:
                print("Podana ścieżka do folderu nie istnieje.")
