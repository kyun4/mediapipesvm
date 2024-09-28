#pip install mediapipe-model-maker
#pip install tensorflow tensorflow-model-optimization

from mediapipe_model_maker import image_classifier
from mediapipe_model_maker.image_classifier import DataLoader

from mediapipe_model_maker import object_detector
from mediapipe_model_maker.object_detector import DataLoader

# Load the dataset
train_data = DataLoader.from_folder('path_to_train_data')
test_data = DataLoader.from_folder('path_to_test_data')

# Create a model using Model Maker (transfer learning from a pre-trained model)
model = image_classifier.create(train_data)

# Evaluate the model on test data
loss, accuracy = model.evaluate(test_data)
print(f'Test accuracy: {accuracy:.4f}')

# Optionally, fine-tune the model for more epochs
model.train(train_data, epochs=5)

# Load the object detection dataset (from a COCO format or CSV)
train_data = DataLoader.from_pascal_voc(image_dir='path_to_images', annotations_dir='path_to_annotations')

# Create an object detection model
model = object_detector.create(train_data)

# Evaluate the model
model.evaluate(test_data)

# Fine-tune the model further
model.train(train_data, epochs=10)

# Export the trained model as a TFLite file
model.export(export_dir='export/')

# Export the model in SavedModel format (for further use in TensorFlow)
model.export(export_dir='export/', export_format='saved_model')

# Or export as a Keras model
model.export(export_dir='export/', export_format='keras')

