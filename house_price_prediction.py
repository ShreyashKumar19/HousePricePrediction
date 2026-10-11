import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("Housing.csv")

print(data.head())

print("\nShape of data:")
print(data.shape)

print("\nMissing values:")
print(data.isnull().sum())

print("\nData description:")
print(data.describe())

# Fill missing values
data = data.fillna(data.median(numeric_only=True))

# Convert text columns into numbers
data = pd.get_dummies(data, drop_first=True)

# Separate features and target
X = data.drop("price", axis=1)
y = data["price"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)

# Show actual and predicted prices
result = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print("\nActual and Predicted Prices:")
print(result.head(10))

# Plot actual and predicted prices
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")

plt.show()

# Take house details from the user
print("\nEnter details of the house to predict its price.")

area = float(input("Enter area (sq ft): "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))
stories = int(input("Enter number of stories: "))

mainroad = input("Is it connected to main road? (like NH or SH) (yes/no): ")
guestroom = input("Does it have a guest room? (yes/no): ")
basement = input("Does it have a basement? (yes/no): ")
hotwaterheating = input("Does the house have a hot water system (such as a geyser)? (yes/no): ")
airconditioning = input("Does it have air conditioning? (yes/no): ")

parking = int(input("Enter number of parking spaces: "))

prefarea = input("Is it in a preferred area? (yes/no): ")
furnishingstatus = input(
    "Enter furnishing status (furnished/semi-furnished/unfurnished): "
)

# Create input data
new_house = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "stories": [stories],
    "mainroad": [mainroad],
    "guestroom": [guestroom],
    "basement": [basement],
    "hotwaterheating": [hotwaterheating],
    "airconditioning": [airconditioning],
    "parking": [parking],
    "prefarea": [prefarea],
    "furnishingstatus": [furnishingstatus]
})

# Convert input data into numbers
new_house = pd.get_dummies(new_house, drop_first=True)

# Make sure input has the same columns as training data
new_house = new_house.reindex(columns=X.columns, fill_value=0)

# Normalize the input
new_house = scaler.transform(new_house)

# Predict the house price
predicted_price = model.predict(new_house)

print("\nPredicted House Price:", predicted_price[0])