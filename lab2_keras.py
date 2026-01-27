import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


X = tf.constant([[8.0, 5.0]])   
y = tf.constant([[64.0]])      


model = Sequential([
    Dense(
        units=1,           
        input_shape=(2,),  
        use_bias=False,       
        activation='linear',
        kernel_initializer=tf.constant_initializer([[2.0], [7.5]])
    )
])


def custom_mse(y_true, y_pred):
    return 0.5 * tf.square(y_pred - y_true)


model.compile(
    optimizer=SGD(learning_rate=0.005),
    loss=custom_mse
)


model.fit(
    X,
    y,
    epochs=100,
    verbose=1  
)


w1, w2 = model.get_weights()[0].flatten()

print("Final weights:")
print("w1 =", w1)
print("w2 =", w2)

print("Final output:")
print(model.predict(X)[0][0])
