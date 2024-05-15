import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("./src/csv/all_stocks_5yr.csv", sep=";")

df = data[data["Name"] == "AAL"]


def plot_predictions(test, predicted):
    plt.plot(test, color="red", label="real stock price")
    plt.plot(predicted, color="blue", label="predicted stock price")
    plt.title("Stock price prediction")
    plt.xlabel("time")
    plt.ylabel("Stock price")
    plt.legend()
    plt.show()


df = data[data["Name"] == "AAL"]

df.loc[:, "date"] = pd.to_datetime(df.loc[:, "date"], format="%d/%m/%Y")

df = df.set_index("date")

df.head()

train = df[:"2016"].iloc[:, 1:2].values
test = df["2017":].iloc[:, 1:2].values

df["close"] = df["close"].astype(float)

df["close"][:"2016"].plot(figsize=(16, 4), legend=True)
df["close"]["2017":].plot(figsize=(16, 4), legend=True)
plt.legend(["Training set (before 2017)", "Test set (from 2017)"])
plt.title("Stock prices")
plt.show()

sc = MinMaxScaler(feature_range=(0, 1))
train_scaled = sc.fit_transform(train)

x_train = []
y_train = []

for i in range(60, 982):
    x_train.append(train_scaled[i - 60 : i, 0])
    y_train.append(train_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

dataset_total = pd.concat((df["close"][:"2016"], df["close"]["2017":]), axis=0)
print(dataset_total.shape)

inputs = dataset_total[len(dataset_total) - len(test) - 60 :].values
print(inputs.shape)
inputs = inputs.reshape(-1, 1)
print(inputs.shape)
inputs = sc.transform(inputs)
print(inputs.shape)

x_test = []
for i in range(60, 311):
    x_test.append(inputs[i - 60 : i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

regressor = Sequential()

regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

regressor.compile(optimizer="rmsprop", loss="mean_squared_error")

regressor.fit(x_train, y_train, epochs=50, batch_size=32)

dataset_total = pd.concat((df["close"][:"2016"], df["close"]["2017":]), axis=0)
print(dataset_total.shape)

inputs = dataset_total[len(dataset_total) - len(test) - 60 :].values
print(inputs.shape)
inputs = inputs.reshape(-1, 1)
print(inputs.shape)
inputs = sc.transform(inputs)
print(inputs.shape)

x_test = []
for i in range(60, 311):
    x_test.append(inputs[i - 60 : i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted = regressor.predict(x_test)
predicted = sc.inverse_transform(predicted)

plot_predictions(test, predicted)
