import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Set your dataset directory path
train_dir = 'train'
test_dir = 'test'

# Define image dimensions and batch size
img_width, img_height = 150, 150
batch_size = 32

# Function to load data using flow_from_directory and store class names
def load_data(train_dir, test_dir):
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Store class names in a file
    class_names = list(train_generator.class_indices.keys())
    with open('class_names.pkl', 'wb') as f:
        pickle.dump(class_names, f)

    return train_generator, test_generator, class_names

# Function to create and train a ResNet50 model
def train_resnet50(train_generator, test_generator, class_names):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(len(class_names), activation='softmax'))

    # Freeze the convolutional base
    base_model.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=100,  # You can adjust the number of epochs
        validation_data=test_generator,
        validation_steps=test_generator.samples // batch_size
    )

    return model, history

# Function to plot accuracy and loss graphs
def plot_graphs(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("accuracy_plot.png")
    plt.show()

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.show()

# Function to save the trained model
def save_model(model, model_name='ResNet50_model'):
    model.save(f'{model_name}.h5')

# Function to evaluate the model and print classification report and confusion matrix
def evaluate_model(model, test_generator):
    Y_pred = model.predict_generator(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(test_generator.classes, y_pred))
    print('Classification Report')
    target_names = list(test_generator.class_indices.keys())
    print(classification_report(test_generator.classes, y_pred, target_names=target_names))

if __name__ == "__main__":
    train_generator, test_generator, class_names = load_data(train_dir, test_dir)
    trained_model, training_history = train_resnet50(train_generator, test_generator, class_names)
    plot_graphs(training_history)
    save_model(trained_model)
    evaluate_model(trained_model, test_generator)
