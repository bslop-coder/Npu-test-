import numpy as np, tensorflow as tf

def conv(model, name, shapes, int8):
    def rep():
        for _ in range(10):
            yield [np.random.randn(*s).astype(np.float32) for s in shapes]
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    if int8:
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = rep
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        c.inference_input_type = tf.int8
        c.inference_output_type = tf.int8
    open(name, "wb").write(c.convert())

mm = tf.keras.layers.Lambda(lambda t: tf.matmul(t[0], t[1]))

a = tf.keras.Input(batch_shape=(16, 1024, 1024))
o = tf.keras.layers.Softmax(axis=-1)(a)
conv(tf.keras.Model(a, o), "sm_int8.tflite", [(16, 1024, 1024)], True)

q = tf.keras.Input(batch_shape=(16, 1024, 128))
kt = tf.keras.Input(batch_shape=(16, 128, 1024))
o = tf.keras.layers.Lambda(lambda t: tf.matmul(t[0], t[1]))([q, kt])
conv(tf.keras.Model([q, kt], o), "qk_int8.tflite", [(16, 1024, 128), (16, 128, 1024)], True)

p = tf.keras.Input(batch_shape=(16, 1024, 1024))
v = tf.keras.Input(batch_shape=(16, 1024, 128))
o = tf.keras.layers.Lambda(lambda t: tf.matmul(t[0], t[1]))([p, v])
conv(tf.keras.Model([p, v], o), "pv_int8.tflite", [(16, 1024, 1024), (16, 1024, 128)], True)

q = tf.keras.Input(batch_shape=(16, 1024, 128))
kt = tf.keras.Input(batch_shape=(16, 128, 1024))
v = tf.keras.Input(batch_shape=(16, 1024, 128))
s = tf.keras.layers.Lambda(lambda t: tf.matmul(t[0], t[1]))([q, kt])
pr = tf.keras.layers.Softmax(axis=-1)(s)
o = tf.keras.layers.Lambda(lambda t: tf.matmul(t[0], t[1]))([pr, v])
conv(tf.keras.Model([q, kt, v], o), "attn_f32.tflite", [(16, 1024, 128), (16, 128, 1024), (16, 1024, 128)], False)
