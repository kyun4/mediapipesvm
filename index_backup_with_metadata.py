from pycocotools.coco import COCO
import cv2
import os

import mediapipe as mp
import cv2
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# FOR GRAPHICAL AND OTHER REPORTS

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

from sklearn.decomposition import PCA
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.model_selection import learning_curve

# EXPORTING MODEL into model.tflite

import tensorflow as tf
from tensorflow import keras

# ADDING METADATA to model.tflite

from tflite_support.metadata import metadata_schema_py_generated as _metadata_fb
from tflite_support.metadata_writers import writer_utils, metadata_writer

# =============== Extracting Images and Labels Data from a made COCO Datasets (From Previous Datasets produced on YoloV8 Model last time) ===================
# Initialize COCO API
coco = COCO('data/labels.json')

# Get images with the hand category (COCO doesn't specifically have 'hand', so you'd manually select or use custom dataset)
image_ids = coco.getImgIds()

# Load an image from COCO
image_data = coco.loadImgs(image_ids[0])[0]
image_path = os.path.join('data/images', image_data['file_name'])
image = cv2.imread(image_path)


# =============== FEATURE EXTRACTION of MEDIAPIPE from HAND LANDMARKS ===================
# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands

# Load an image (assuming 'image' is loaded as a NumPy array)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image to detect hands and extract landmarks
with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
    result = hands.process(image_rgb)

# Extract landmarks for each hand detected in the image
if result.multi_hand_landmarks:
    for hand_landmarks in result.multi_hand_landmarks:
        # Extract the 21 hand landmarks (x, y, z coordinates)
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        print("Hand landmarks: ", landmarks)  # This will print the hand features


# Prepare feature vectors and corresponding labels
features = []  # Store hand landmark features (flattened)
labels = []  # Store labels (e.g., COCO object category or custom class labels)

for img_id in image_ids:
    img_data = coco.loadImgs(img_id)[0]
    img_path = os.path.join('data/images', img_data['file_name'])
    image = cv2.imread(img_path)
    
    # Process the image with MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
        result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract and flatten the landmarks
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            features.append(landmarks.flatten())  # Flatten 21x3 matrix into a vector of 63

            # Assign labels (e.g., object category ID or custom label)
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            if anns:
                labels.append(anns[0]['category_id'])  # Assuming using COCO category IDs


# Convert features and labels into numpy arrays
features = np.array(features)
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============== SVM TRAINING ===================
# Train the SVM classifier
svm = SVC(kernel='linear')  # You can try other kernels like 'rbf'
svm.fit(X_train, y_train)


# =============== GRAPHS and REPORTS (SVM Accuracy, Precision Report, Confusion Matrix, Learning Curve and SVM Decision Boundary) ===================
# Evaluate the SVM model
accuracy = svm.score(X_test, y_test)
print(f"SVM Accuracy: {accuracy}")


y_pred = svm.predict(X_test)

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using Seaborn for heatmap visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

report = classification_report(y_test, y_pred)
print(report)


# Reduce features to 2D using PCA for visualization
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)

# Train an SVM on the reduced 2D data
svm_2d = SVC(kernel='linear')
svm_2d.fit(X_train_2d, y_train)

# Plot the decision boundary
plt.figure(figsize=(10, 8))

# Create a meshgrid for plotting decision boundary
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict on the meshgrid points
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, s=50, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('SVM Decision Boundary in 2D using PCA')
plt.show()

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


# Generate learning curve data
train_sizes, train_scores, test_scores = learning_curve(svm, X_train, y_train, cv=5)

# Calculate mean and standard deviation for plotting
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label="Training score", color="blue")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.2)
plt.plot(train_sizes, test_mean, label="Cross-validation score", color="green")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="green", alpha=0.2)
plt.title('Learning Curve')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.legend(loc='best')
plt.show()

# =============== EXPORT model.tflite ===================
# Example neural network model
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(63,)),  # Assuming hand landmarks with 63 features
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(10, activation='softmax')  # Assuming 10 classes for classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model as a .tflite file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# Save the quantized model
with open('model_quant.tflite', 'wb') as f:
    f.write(tflite_quant_model)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on a sample input
input_data = X_test[0:1].astype(np.float32)  # Assuming X_test is your test dataset
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the prediction
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Post ML Prediction:", output_data)

# =============== FOR METADATA ===================
# Define input/output information
input_meta = metadata_writer.TensorMetadata()
input_meta.name = "input_tensor"
input_meta.description = "Input image to be classified."
input_meta.content = metadata_writer.Content(input_dtype=_metadata_fb.TensorType.FLOAT32)
input_meta.shape = [1, 224, 224, 3]  # Assuming an image input of size 224x224 with 3 channels (RGB)

output_meta = metadata_writer.TensorMetadata()
output_meta.name = "output_tensor"
output_meta.description = "The predicted label for the input image."
output_meta.content = metadata_writer.Content(output_dtype=_metadata_fb.TensorType.FLOAT32)
output_meta.shape = [1, 10]  # Assuming a classification output with 10 possible classes

# Define associated labels (optional)
output_meta.associated_files = ["labels.txt"]

# Create the model metadata
model_meta = metadata_writer.ModelMetadata()
model_meta.name = "My Image Classifier"
model_meta.version = "v1.0"
model_meta.description = "A TensorFlow Lite model for image classification"
model_meta.input_metadata.append(input_meta)
model_meta.output_metadata.append(output_meta)

# Path to the TFLite model
model_file = 'model.tflite'

# Create metadata buffer
model_metadata_buffer = model_meta.get_serialized_metadata()

# Load the TFLite model
with open(model_file, "rb") as f:
    tflite_model = f.read()

# Save the model with metadata
with open("model_with_metadata.tflite", "wb") as f:
    # Add metadata to the model file
    f.write(writer_utils.ModelWithMetadata(model_buffer=tflite_model, metadata_buffer=model_metadata_buffer).save())

# You can also attach associated files like label maps
writer_utils.attach_associated_file("model_with_metadata.tflite", ["labels.txt"])

input_meta.process_units.append(
    metadata_writer.NormalizationProcessUnit(
        mean=[127.5], std=[127.5], description="Scale image from [0, 255] to [-1, 1]"
    )
)