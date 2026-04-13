--------------------------------------------------------------Practical - 1-----------------------------------------------------------------------------------------------
import numpy as np 
import tensorflow as tf

num = np.array([[1, 2], [3, 4]])
eigen_values, eigen_vector = np.linalg.eig(num)

print("Eigen value of given array:",eigen_values)
print("Eigen vector of given array:",eigen_vector)

num2 = np.array([[1, 2, 3],
                [2, 3, 4],
                [5, 3, 4]])

eig_values, eig_vect = np.linalg.eig(num2)

print("Eigen values of the given array:", eig_values)
print("Eigen vector of the given array:", eig_vect)

# In Tensorflow 
e_matrix = tf.random.uniform([2,2], minval=3, maxval=10, dtype=float)
print("Matrix A: \n{}\n\n".format(e_matrix))

e_values, e_vector = tf.linalg.eig(e_matrix)
print("Eigen Values: \n{}\n\n Eigen Vector: \n{}\n".format(e_values, e_vector))

--------------------------------------------------------------Practical - 2--------------------------------------------------------------------------------------------
import numpy as np

def unitstep(v):
    if v >= 0:
        return 1
    else:
        return 0

def perceptronModel(x, w, b):
    v = np.dot(x, w) + b
    y = unitstep(v)
    return y

def NOTlogicfunct(x):
    wNOT = -1
    bAND = 0.5
    return perceptronModel(x, wNOT, bAND)

def AND_logicfunct(x):
    w = np.array([1, 1])
    bAND = -1.5
    return perceptronModel(x, w, bAND)

def OR_logicfunct(x):
    w = np.array([1, 1])
    bOR = -0.5
    return perceptronModel(x, w, bOR)

def XOR_logicfunct(x):
    y1 = AND_logicfunct(x)
    y2 = OR_logicfunct(x)
    y3 = NOTlogicfunct(y1)
    final_x = np.array([y2, y3])
    final_output = AND_logicfunct(final_x)
    y3 = NOTlogicfunct(y1)
    return final_output

test1 = np.array([0,1])
test2 = np.array([1,1])
test3 = np.array([0,0])
test4 = np.array([1,0])

print("XOR ({}, {}) = {}". format(0, 1, XOR_logicfunct(test1)))
print(f"XOR (1, 1) = {XOR_logicfunct(test2)}")
print("XOR ({}, {}) = {}". format(0, 1, XOR_logicfunct(test3)))
print("XOR ({}, {}) = {}". format(0, 1, XOR_logicfunct(test4)))

-----------------------------------------------------------------Practical - 3--------------------------------------------------------------------------------------------
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Dropout

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape((-1, 28*28))
x_test = x_test.reshape((-1, 28*28))

model = tf.keras.Sequential([
    Input(shape=(784,)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

histroy = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(histroy.history['accuracy'], label='Training accuracy')
plt.plot(histroy.history['val_accuracy'], label='validation_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

plt.subplot(1, 2, 2)
plt.plot(histroy.history['loss'], label='Training loss')
plt.plot(histroy.history['val_loss'], label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

predictions = model.predict(x_test[:5])
print("Prediction shape:", predictions.shape)
print("Prediction classes:", np.argmax(predictions, axis=1))
print("Actual classes:  ", y_test[:5])

plt.figure(figsize=(10,2))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'Pred: {np.argmax(predictions[i])}\nTrue: {y_test[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()
----------------------------------------------------------------Practical - 4-----------------------------------------------------------------------------------------
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=5,
    validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

prediction = model.predict(x_test)

plt.figure(figsize=(15, 4))
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.axis("off")
    plt.title("Label: " + str(np.argmax(y_test[i])))

    plt.subplot(2, 10, i+11)
    plt.imshow(x_test[i]. reshape(28,28), cmap='gray')
    plt.axis('off')
    plt.title("Pred: " + str(np.argmax(prediction[i])))
plt.show()
----------------------------------------------------------------Practical - 5------------------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def create_time_series_data():
    time = np.arange(0, 100, 0.1)
    series = np.sin(time) + np.random.normal(0, 0.1, len(time))
    series = series + time / 50
    return time, series

def create_dataset(data, timesteps=10):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i:(i + time_steps)])
    return np.array(X), np.array(y)

time, series = create_time_series_data()

scaler = MinMaxScaler(feature_range=(0,1))
series_scaled = scaler.fit_transform(series.reshape(-1, 1))

time_steps = 20
X, y = create_dataset(series_scaled, time_steps)

x = X.reshape(X.shape[0], X.shape[1], 1)

train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(50, activation='relu', input_shape=(time_steps, 1)),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer='adam', loss='mse'
)
model.summary()

history = model.fit(
    x_train, y_train, 
    epochs=50, 
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

train_prediction = model.predict(x_train)
test_prediction = model.predict(x_test)

train_prediction_reshaped = train_prediction.reshape(train_prediction.shape[0], -1)
test_prediction_reshaped = test_prediction.reshape(test_prediction.shape[0], -1)

y_train_reshaped = y_train.reshape(y_train.shape[0], -1) if y_train.ndim > 2 else y_train
y_test_reshaped = y_test.reshape(y_test.shape[0], -1) if y_test.ndim > 2 else y_test

train_predict = scaler.inverse_transform(train_prediction_reshaped)
y_train_inv = scaler.inverse_transform(y_train_reshaped)

test_predict = scaler.inverse_transform(test_prediction_reshaped)
y_test_inv = scaler.inverse_transform(y_test_reshaped)

train_rmse = np.sqrt(np.mean((train_predict - y_train_inv) ** 2))
test_rmse = np.sqrt(np.mean((test_predict - y_test_inv) **2))
print(f'Train RMSE: {train_rmse:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')

plt.figure(figsize=(12, 6))
plt.plot(time, series, label='Original Series', alpha=0.5)

train_plot_idx = np.arange(time_steps, time_steps + len(train_predict))
plt.plot(time[train_plot_idx], train_predict, 'b', label='Trianing Predictions')

test_plot_idx = np.arange(time_steps +len(train_predict), time_steps + len(train_predict) + len(test_predict))
plt.plot(time[test_plot_idx], test_predict, 'r', label='Test Predictions')

plt.title('RNN time series prediction')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

def predict_sequence(model, first_batch, n_steps):
    curr_batch = first_batch.copy()
    predicted = []

    for _ in range(n_steps):
        curr_pred = model.predict(curr_batch)[0][0]
        predicted.append(curr_pred)
        
        curr_batch = np.roll(curr_batch, -1, axis=1)
        curr_batch[0,-1,0] = curr_pred
        
    return np.array(predicted)

future_steps = 100
last_batch = x_test[-1].reshape(1, time_steps, 1)

future_pred = predict_sequence(model, last_batch, future_steps)
future_pred = scaler.inverse_transform(future_pred.reshape(-1,1))

future_time = np.arange(time[-1], time[-1] + future_steps * 0.1, 0.1)

plt.figure(figsize=(14, 6))
plt.plot(time, series, label='Original Series', alpha=0.5)
plt.plot(time[test_plot_idx], test_predict, 'r', label='Test Predictions')
plt.plot(future_time, future_pred, 'g', label='Future Predictions')
plt.axvline(x=time[-1], color='k', linestyle='--')
plt.title('RNN Time Series Prediction with Future Forecast')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
---------------------------------------------------------------Practical - 6----------------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

vocab_size = 10000
maxlen = 200

(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=vocab_size)

mask = np.array([len(x) > 0 for x in x_train])
x_train = x_train[mask]
y_train = y_train[mask]

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=maxlen),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(x_train, y_train, epochs=3, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(x_test, y_test)
print(f"\n✅ Test Accuracy: {accuracy:.4f}")

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

def decode_review(encoded):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded])

print("\nSample decoded review:")
print(decode_review(x_train[0]))
print("\nSentiment:", "Positive" if y_train[0] == 1 else "Negative")
----------------------------------------------------------------Practical - 7---------------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# 1. Load and preprocess MNIST data
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 127.5 - 1  # normalize to [-1, 1]
x_train = x_train.reshape(-1, 28, 28, 1)

BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 50
NOISE_DIM = 100

train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))

    return model

def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator = build_generator()
discriminator = build_discriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

def generate_and_plot_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0]*127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.suptitle(f"Epoch {epoch}")
    plt.show()

seed = tf.random.normal([16, NOISE_DIM])

def train(dataset, epochs):
    for epoch in range(1, epochs+1):
        for image_batch in dataset:
            train_step(image_batch)

        print(f"Epoch {epoch} completed.")
        generate_and_plot_images(generator, epoch, seed)

# 8. Start training
train(train_dataset, EPOCHS)
