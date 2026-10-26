import numpy as np, tensorflow as tf

def make(tokens, din, dout):
    m = tf.keras.Sequential([
        tf.keras.layers.InputLayer(batch_input_shape=(1, tokens, 1, din)),
        tf.keras.layers.Conv2D(dout, 1),
    ])
    def rep():
        for _ in range(10):
            yield [np.random.randn(1, tokens, 1, din).astype(np.float32)]
    c = tf.lite.TFLiteConverter.from_keras_model(m)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = rep
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8
    c.inference_output_type = tf.int8
    open("conv_%d_%d_%d.tflite" % (tokens, din, dout), "wb").write(c.convert())

make(256, 2048, 8192)
make(1024, 2048, 2048)
