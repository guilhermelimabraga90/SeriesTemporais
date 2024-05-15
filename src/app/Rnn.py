import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, SimpleRNN
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("./src/csv/all_stocks_5yr.csv", sep=";")

df = data[data["Name"] == "AAL"]

train_length = round(len(df) * 0.7)
lg = len(df)
val_length = lg - train_length

train_data = df["close"][:train_length,]
val_data = df["close"][train_length:,]

train = train_data.values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_trainset = scaler.fit_transform(train)

x_train = []
y_train = []
step = 50

for i in range(step, train_length):
    x_train.append(scaled_trainset[i - step : i, 0])
    y_train.append(scaled_trainset[i, 0])

X_train, y_train = np.array(x_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
y_train.reshape(y_train.shape[0], 1)

model = Sequential()

model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))

model.add(Dropout(0.2))

model.add(SimpleRNN(units=50, return_sequences=True))

model.add(Dropout(0.2))

model.add(SimpleRNN(units=50, return_sequences=True))

model.add(Dropout(0.2))

model.add(SimpleRNN(units=50))

model.add(Dropout(0.2))

model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=50, batch_size=32)

y_pred = model.predict(X_train)
y_pred = scaler.inverse_transform(y_pred.reshape(1, -1))

y_train = scaler.inverse_transform(y_train.reshape(1, -1))
y_train

y_train.shape
y_train = np.reshape(y_train, (831, 1))

y_pred.shape
y_pred = np.reshape(y_pred, (831, 1))

plt.figure(figsize=(30, 10))
plt.plot(y_pred, ls="--", label="y_pred", lw=2)
plt.plot(y_train, label="y_train")
plt.legend()
plt.show()

val = val_data.values.reshape(-1, 1)

scaled_valset = scaler.fit_transform(val)

xval_train = []
yval_train = []
step = 50

for i in range(step, val_length):
    xval_train.append(scaled_valset[i - step : i, 0])
    yval_train.append(scaled_valset[i, 0])

X_val, y_val = np.array(xval_train), np.array(yval_train)

X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
y_val = np.reshape(y_val, (-1, 1))

y_pred_val = model.predict(X_val)

y_pred_val = scaler.inverse_transform(y_pred_val)

y_val_is = scaler.inverse_transform(y_val)

plt.figure(figsize=(30, 10))
plt.plot(y_pred_val, label="y_pred")
plt.plot(y_val_is, label="y_val")
plt.legend()
plt.show()
