import tensorflow as tf
m = tf.keras.models.load_model('models/emotion_model.keras')
print('MODEL OUTPUT SHAPE:', m.output_shape)
print('NUMBER OF OUTPUTS:', m.output_shape[-1] if m.output_shape else None)
print('\nFINAL LAYERS:')
for layer in m.layers[-5:]:
    try:
        print(layer.name, getattr(layer, 'output_shape', 'n/a'))
    except Exception:
        print(layer.name, 'n/a')

print('\nSUMMARY:')
m.summary()
