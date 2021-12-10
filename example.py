from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split

from nn import myNN

df = pd.read_csv(r"C:\Users\User\Desktop\deep_learning\ins.csv")
print(df.head)

x_train, x_test, y_train, y_test = train_test_split(df[['age','affordability']], df.bought_insurance,
                                                    test_size=0.2, random_state=25)

x_train_scaled = x_train.copy()
x_train_scaled['age'] = x_train_scaled['age'] / 100

x_test_scaled = x_test.copy()
x_test_scaled = x_test_scaled / 100

model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(2,), activation='sigmoid', kernel_initializer='ones', bias_initializer='zeros')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_scaled, y_train, epochs=10000)

model.evaluate(x_test_scaled, y_test)

model.predict(x_test_scaled)

coef, intercept = model.get_weights()
print(coef, intercept)


if __name__ == '__main__':
    customModel = myNN()
    customModel.fit(x_train_scaled, y_train, epochs=8000, loss_threshold=0.4631)
    print(coef, intercept)
