import numpy as np
import tensorflow as tf
from tensorflow.python import keras

import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='from which to generate TFRecord, folders or mxrec', default='mxrec')
    parser.add_argument('--image_size', type=int, help='image size', default=112)
    parser.add_argument('--read_dir', type=str, help='directory to read data', default='')
    parser.add_argument('--save_path', type=str, help='path to save TFRecord file', default='')
    parser.add_argument('--thread_num', type=int, help='number of thread to progress data', default=None)

    return parser.parse_args()


# Load the MobileNet tf.keras model.
assert int(tf.__version__.split(".")[0]) >= 2, "tensorflow{} 版本不满足".format(tf.__version__)
with open("model.json") as json_file:
    json_config = json_file.read()
model = keras.models.model_from_json(json_config)
model.load_weights('weights.h5')

out_path = "./demo.tflite"

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(out_path, "wb") as f:
    f.write(tflite_model)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the TensorFlow Lite model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
tflite_results = interpreter.get_tensor(output_details[0]['index'])

# Test the TensorFlow model on random input data.
tf_results = model(tf.constant(input_data))
print(tf_results)
# Compare the result.
for tf_result, tflite_result in zip(tf_results, tflite_results):
  np.testing.assert_almost_equal(tf_result, tflite_result, decimal=5)