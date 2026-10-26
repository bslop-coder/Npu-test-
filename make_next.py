import numpy as np, tensorflow as tf

def conv(m, name, shape, mode):
    def rep():
        for _ in range(20):
            yield [np.random.randn(*shape).astype(np.float32)]
    c = tf.lite.TFLiteConverter.from_keras_model(m)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = rep
    if mode == "int8":
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        c.inference_input_type = tf.int8
        c.inference_output_type = tf.int8
    else:
        c.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    open(name, "wb").write(c.convert())

big = tf.keras.Sequential([
    tf.keras.layers.InputLayer(batch_input_shape=(1024, 2048)),
    tf.keras.layers.Dense(8192, activation="relu"),
    tf.keras.layers.Dense(2048),
])
conv(big, "big_int8.tflite", (1024, 2048), "int8")

small = tf.keras.Sequential([
    tf.keras.layers.InputLayer(batch_input_shape=(256, 512)),
    tf.keras.layers.Dense(2048, activation="relu"),
    tf.keras.layers.Dense(512),
])
try:
    conv(small, "lin_w8a16.tflite", (256, 512), "w8a16")
except Exception as e:
    print("w8a16 failed", e)

q = tf.keras.Input(batch_shape=(1, 16, 256, 128))
k = tf.keras.Input(batch_shape=(1, 16, 256, 128))
v = tf.keras.Input(batch_shape=(1, 16, 256, 128))
s = tf.keras.layers.Lambda(lambda t: tf.matmul(t[0], t[1], transpose_b=True))([q, k])
p = tf.keras.layers.Softmax(axis=-1)(s)
o = tf.keras.layers.Lambda(lambda t: tf.matmul(t[0], t[1]))([p, v])
att = tf.keras.Model([q, k, v], o)

def rep3():
    for _ in range(20):
        yield [np.random.randn(1, 16, 256, 128).astype(np.float32) for _ in range(3)]

c = tf.lite.TFLiteConverter.from_keras_model(att)
c.optimizations = [tf.lite.Optimize.DEFAULT]
c.representative_dataset = rep3
c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
c.inference_input_type = tf.int8
c.inference_output_type = tf.int8
open("attn_int8.tflite", "wb").write(c.convert())
