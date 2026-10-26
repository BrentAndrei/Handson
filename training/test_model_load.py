import tensorflow as tf
model_path = "model_out/fsl_model.h5"
model = tf.keras.models.load_model(model_path)
print(f"Model loaded. Number of parameters: {model.count_params()}")
