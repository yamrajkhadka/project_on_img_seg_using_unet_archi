"""Export the paper's Keras U-Net to ONNX with a dynamic image size, and verify it.

The saved model has a fixed 224 x 224 input.  The network is fully convolutional, so we rebuild the identical
architecture with an (None, None, 3) input, copy the weights, and export that; any H, W that are multiples of 16
work.  The script checks the export against Keras at 224 and 448 px.

Usage:
    python scripts/convert_to_onnx.py deepglobe_unet_jay.keras models/unet_deepglobe.onnx
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402


def build_unet(input_shape=(None, None, 3), num_classes=7):
    """Identical to build_unet() in DeepGlobe_LandCoverSegmentation_U-Net_yamraj.ipynb."""
    from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dropout, Input, MaxPooling2D, concatenate
    from keras.models import Model

    def conv_block(x, filters):
        x = Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        return x

    inputs = Input(input_shape)
    c1 = conv_block(inputs, 32); p1 = MaxPooling2D()(c1)  # noqa: E702
    c2 = conv_block(p1, 64); p2 = MaxPooling2D()(c2)  # noqa: E702
    c3 = conv_block(p2, 128); p3 = MaxPooling2D()(c3)  # noqa: E702
    c4 = conv_block(p3, 256); p4 = MaxPooling2D()(c4)  # noqa: E702
    b = conv_block(p4, 512); b = Dropout(0.5)(b)  # noqa: E702
    u1 = Conv2DTranspose(256, 2, strides=2, padding="same")(b); c5 = conv_block(concatenate([u1, c4]), 256)  # noqa: E702
    u2 = Conv2DTranspose(128, 2, strides=2, padding="same")(c5); c6 = conv_block(concatenate([u2, c3]), 128)  # noqa: E702
    u3 = Conv2DTranspose(64, 2, strides=2, padding="same")(c6); c7 = conv_block(concatenate([u3, c2]), 64)  # noqa: E702
    u4 = Conv2DTranspose(32, 2, strides=2, padding="same")(c7); c8 = conv_block(concatenate([u4, c1]), 32)  # noqa: E702
    return Model(inputs, Conv2D(num_classes, 1, activation="softmax")(c8))


def main(src: str, dst: str) -> None:
    import keras
    import onnx
    import onnxruntime as ort
    import tensorflow as tf
    import tf2onnx

    fixed = keras.saving.load_model(src, compile=False)
    dyn = build_unet()
    dyn.set_weights(fixed.get_weights())
    spec = (tf.TensorSpec((None, None, None, 3), tf.float32, name="image"),)

    @tf.function(input_signature=spec)
    def serve(image):
        return {"probs": dyn(image, training=False)}

    proto, _ = tf2onnx.convert.from_function(serve, input_signature=spec, opset=17)
    meta = proto.metadata_props.add()
    meta.key, meta.value = "description", ("U-Net land-cover segmentation (DeepGlobe, 7 classes), Khadka & Pradhan "
                                           "(2026) doi:10.3126/joeis.v5i2.97825. NHWC float32 RGB in [0,1].")
    onnx.checker.check_model(proto)
    onnx.save(proto, dst)

    sess = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    for size in (224, 448):
        x = rng.random((2, size, size, 3), dtype=np.float32)
        ref = (fixed if size == 224 else dyn).predict(x, verbose=0)
        out = sess.run(None, {"image": x})[0]
        print(f"{size}px: max |onnx - keras| = {np.abs(out - ref).max():.2e}, "
              f"argmax agreement = {(out.argmax(-1) == ref.argmax(-1)).mean():.6f}")
    print(f"wrote {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
