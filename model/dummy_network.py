import os
import sys
import tensorflow as tf
import numpy as np
import onnx
import keras2onnx

def build_network(input_dim, output_dim, hidden_layers):
    input_layer = tf.keras.Input(input_dim)
    a = tf.keras.layers.Dense(hidden_layers[0],activation='relu')(input_layer)

    for i in range(1,len(hidden_layers)):
        a = tf.keras.layers.Dense(hidden_layers[i],activation='relu')(a)

    output_layer = tf.keras.layers.Dense(output_dim)(a)

    model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer], name="dummy_network")
    
    model.summary()
    
    return model


if len(sys.argv) != 2:
    print("Usage: {} <output path for onnx file>".format(sys.argv[0]))
    exit(1);

assert len(sys.argv[1]) > 0
print("Save models to '" + sys.argv[1] + "'")

input_dim = 5
output_dim = 5

model = build_network(input_dim,output_dim,[500,500,500])

# Onnx
onnx_model = keras2onnx.convert_keras(model, model.name)
onnx.save_model(onnx_model, os.path.join(sys.argv[1], "dummy_network.onnx"))

print(model(np.full(shape=(input_dim,5), fill_value=1.0)))
