import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

data = pd.read_csv(Path(__file__).parent / "car_data.csv")
print(data.head())
print(data)

filtered_data = data[data["year"] == 2015]
print("Filtered Data for Year 2015:")
print(filtered_data)

plt.scatter(data["mileage_km"], data["price_usd"])
plt.title("Car Mileage vs Price")
plt.xlabel("Mileage (km)")
plt.ylabel("Price")
plt.show()