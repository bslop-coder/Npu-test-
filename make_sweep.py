import numpy as np, tensorflow as tf

def make(tokens, din, dout):
    m = tf.keras.Sequential([
        tf.keras.layers.InputLayer(batch_input_shape=(tokens, din)),
        tf.keras.layers.Dense(dout),
    ])
    def rep():
        for _ in range(10):
            yield [np.random.randn(tokens, din).astype(np.float32)]
    c = tf.lite.TFLiteConverter.from_keras_model(m)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = rep
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8
    c.inference_output_type = tf.int8
    open("fc_%d_%d_%d.tflite" % (tokens, din, dout), "wb").write(c.convert())

make(256, 2048, 2048)
make(256, 2048, 8192)
make(1024, 2048, 2048)
make(64, 2048, 8192)
make(256, 512, 2048)
