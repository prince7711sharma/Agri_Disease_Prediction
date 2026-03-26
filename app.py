import tensorflow as tf

model = tf.keras.models.load_model("model/plant_disease_model.keras")
print("MODEL OK")