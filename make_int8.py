import numpy as np, tensorflow as tf

m = tf.keras.Sequential([
    tf.keras.layers.InputLayer(batch_input_shape=(256, 512)),
    tf.keras.layers.Dense(2048, activation="relu"),
    tf.keras.layers.Dense(512),
])

def rep():
    for _ in range(20):
        yield [np.random.randn(256, 512).astype(np.float32)]

c = tf.lite.TFLiteConverter.from_keras_model(m)
c.optimizations = [tf.lite.Optimize.DEFAULT]
c.representative_dataset = rep
c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
c.inference_input_type = tf.int8
c.inference_output_type = tf.int8
open("lin_int8.tflite", "wb").write(c.convert())

c2 = tf.lite.TFLiteConverter.from_keras_model(m)
c2.optimizations = [tf.lite.Optimize.DEFAULT]
c2.target_spec.supported_types = [tf.float16]
open("lin_fp16.tflite", "wb").write(c2.convert())
