import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='./test_images/cautleya_spicata.jpg', action="store", type = str, help='Image path')
parser.add_argument('--model', default='./model1.h5', action="store", type = str, help='Classifier path')
parser.add_argument('--top_k', default=5, action="store", type=int, help='Return the top K most likely classes')
parser.add_argument('--category_names', default='./label_map.json', action="store", type=str, help='Path to a JSON file mapping labels to flower names')

arg_parser = parser.parse_args()

image_path = arg_parser.input
model_path = arg_parser.model
top_k = arg_parser.top_k
category_names = arg_parser.category_names

batch_size = 32
image_size = 224


class_names = {}

def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()
    

def predict(image_path, model, top_k):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    image = process_image(test_image)
    expanded_image = np.expand_dims(image, axis=0)
    probes = model.predict(expanded_image, verbose=0)
    top_k_values, top_k_index = tf.nn.top_k(probes, k=top_k)
    
    top_k_values = top_k_values.numpy()
    top_k_index = top_k_index.numpy()
    
    return top_k_values[0], top_k_index[0]

if __name__== "__main__":
    with open(category_names, 'r') as f:
        class_names = json.load(f)

    reloaded_keras_model = tf.keras.models.load_model(model_path, custom_objects = {'KerasLayer':hub.KerasLayer}, compile = False)

    probs, labels = predict(image_path, reloaded_keras_model, top_k)

    print ("\nTop {} Classes\n".format(top_k))

    for i, prob, label in zip(range(1, top_k+1), probs, labels):
        print(i)
        print('Label:', label)
        print('Class name:', class_names[str(label+1)].title())
        print('Probability:', prob)
        print('----------')