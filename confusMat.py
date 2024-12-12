import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
MODEL_PATH = 'pokemon_classifier_final.keras'  # Replace with your model path
model = load_model(MODEL_PATH)

# Load the class indices mapping
CLASS_INDICES_PATH = 'class_indices.json'  # Replace with your saved class indices path
with open(CLASS_INDICES_PATH, 'r') as f:
    class_indices = json.load(f)

# Reverse the class indices to map from index to Pokémon name
class_labels = {v: k for k, v in class_indices.items()}

# Function to preprocess an image
def preprocess_image(image_path, target_size=(150, 150)):
    """
    Preprocess the image to match the input size of the model.
    """
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict a Pokémon class
def predict_pokemon(image_path):
    """
    Predict the class of a Pokémon image and return the predicted label.
    """
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)[0]
    top_index = np.argmax(predictions)
    predicted_label = class_labels[top_index]
    return predicted_label

# Function to load test data from a directory structure
def load_test_data(test_dir):
    """
    Load test data and labels from directory structure.
    Args:
        test_dir (str): Path to the test data directory.
    
    Returns:
        tuple: (list of true labels, list of image paths)
    """
    true_labels = []
    image_paths = []
    
    for label_folder in os.listdir(test_dir):
        label_path = os.path.join(test_dir, label_folder)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Acceptable image formats
                    true_labels.append(label_folder)
                    image_paths.append(os.path.join(label_path, img_file))
    
    return true_labels, image_paths

# Function to compute and display the confusion matrix
def compute_and_display_confusion_matrix(true_labels, image_paths):
    """
    Compute and display the confusion matrix.
    Args:
        true_labels (list): List of true labels corresponding to image paths.
        image_paths (list): List of image file paths to be predicted.
    """
    predicted_labels = [predict_pokemon(img) for img in image_paths]
    true_indices = [class_indices[label] for label in true_labels]
    predicted_indices = [class_indices[label] for label in predicted_labels]

    # Compute confusion matrix
    cm = confusion_matrix(true_indices, predicted_indices)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_indices.keys()))
    
    # Plot and display confusion matrix
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()

# Example usage
if __name__ == "__main__":
    # Path to the test data directory
    TEST_DIR = 'dataset/validation'  # Replace with your test data directory path
    
    # Load test data
    true_labels, image_paths = load_test_data(TEST_DIR)

    # Limit to 100 samples
    MAX_SAMPLES = 100
    true_labels = true_labels[:MAX_SAMPLES]
    image_paths = image_paths[:MAX_SAMPLES]

    # Display the confusion matrix
    compute_and_display_confusion_matrix(true_labels, image_paths)
