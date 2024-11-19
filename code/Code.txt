!pip install gradio
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.metrics import Precision, Recall
import matplotlib.pyplot as plt
from google.colab import drive
import gradio as gr

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Define paths
train_path = "/content/drive/MyDrive/project/Newfolder/Split_smol/train"
validation_path = "/content/drive/MyDrive/project/Newfolder/Split_smol/val"
model_save_path = "/content/drive/MyDrive/savedmodel/skin_disease_model.h5"

# DataLoader Class
class DataLoader:
    def __init__(self, img_size=(450, 450), batch_size=25):
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_generator = None
        self.validation_generator = None

    def preprocess_data(self, path):
        datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
        )
        generator = datagen.flow_from_directory(
            path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            color_mode="rgb"
        )
        return generator

    def load_data(self, train_path, validation_path):
        self.train_generator = self.preprocess_data(train_path)
        self.validation_generator = self.preprocess_data(validation_path)
        return self.train_generator, self.validation_generator

# CNNModel Class
class CNNModel:
    def __init__(self, img_size=(450, 450), num_classes=9):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        self.model = Sequential([
            Conv2D(64, (3, 3), padding='same', input_shape=(self.img_size[0], self.img_size[1], 3), kernel_initializer='he_normal'),
            LeakyReLU(alpha=0.15),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            Conv2D(128, (5, 5), padding='same', kernel_initializer='he_normal'),
            LeakyReLU(alpha=0.15),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            Conv2D(128, (5, 5), padding='same', kernel_initializer='he_normal'),
            LeakyReLU(alpha=0.15),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'),
            LeakyReLU(alpha=0.15),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal'),
            LeakyReLU(alpha=0.15),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            Flatten(),
            Dense(256),
            LeakyReLU(alpha=0.15),
            BatchNormalization(),
            Dropout(0.4),
            Dense(256),
            LeakyReLU(alpha=0.15),
            BatchNormalization(),
            Dropout(0.4),
            Dense(self.num_classes, activation='softmax')
        ])
        return self.model

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])
        print("Model compiled successfully.")

    def train_model(self, train_generator, validation_generator, epochs=20):  # Updated to 20 epochs
        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs
        )
        return history

    def save_model(self, filepath):
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

# Check if model exists, otherwise train and save
if not os.path.exists(model_save_path):
    print("Model file not found. Training and saving the model.")
    data_loader = DataLoader()
    train_gen, val_gen = data_loader.load_data(train_path, validation_path)

    cnn_model = CNNModel()
    model = cnn_model.build_model()
    cnn_model.compile_model()
    history = cnn_model.train_model(train_gen, val_gen, epochs=20)  # Updated to 20 epochs
    cnn_model.save_model(model_save_path)
else:
    print("Loading the saved model.")
    model = load_model(model_save_path)

# Define Gradio interface
class_labels = ['Disease1', 'Disease2', 'Disease3', 'Disease4', 'Disease5', 'Disease6', 'Disease7', 'Disease8', 'Disease9']  # Replace with actual class names

def predict_skin_disease(img):
    img = img.resize((450, 450))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)
    return f"{predicted_class} ({confidence:.2f})"

# Create Gradio interface
interface = gr.Interface(fn=predict_skin_disease, inputs=gr.Image(type="pil"), outputs="text", title="Skin Disease Prediction")

# Launch the interface
interface.launch(debug=True)
