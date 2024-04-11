import keras
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_csv('/Users/chrisflannery/python/machine_learning/housing_prices/IowaHousingPrices.csv')
squareFeet = df[['SquareFeet']].values
salePrice = df[['SalePrice']].values

model = keras.Sequential()
model.add(keras.layers.Dense(1, input_shape=(1,)))
model.compile(keras.optimizers.Adam(learning_rate=1.0), 'mean_squared_error')

model.fit(squareFeet,salePrice, epochs=30, batch_size=10)

df.plot(kind='scatter', x='SquareFeet', y='SalePrice', title="Housing Prices and Square Footage of Iowa Homes")
y_pred = model.predict(squareFeet)
plt.plot(squareFeet, y_pred, color="red")
plt.show()

newSF = 2000.0
print(model.predict(x=np.array([newSF])))