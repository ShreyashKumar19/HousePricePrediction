# House Price Prediction

This is my third machine learning project for my online internship.

## Objective

The aim of this project is to predict the price of a house using different house features.

## Dataset

I used a housing price dataset containing 545 house records.

The dataset contains features such as:

- Area
- Bedrooms
- Bathrooms
- Stories
- Main road
- Guest room
- Basement
- Hot water heating
- Air conditioning
- Parking
- Preferred area
- Furnishing status
- Price

The dataset is downloaded using `download_dataset.py`.

## Libraries Used

- Python
- Pandas
- Matplotlib
- Scikit-learn

## Steps

1. Download the dataset.
2. Load the dataset.
3. Explore the data.
4. Check for missing values.
5. Fill missing values.
6. Convert text columns into numbers.
7. Separate features and target.
8. Split the data into training and testing sets.
9. Normalize the features.
10. Train a Linear Regression model.
11. Predict house prices.
12. Evaluate the model using Mean Squared Error and R2 Score.
13. Plot actual and predicted prices.

## How to Run

First install the required libraries:

```bash
pip install -r requirements.txt
```

Download the dataset:

```bash
python download_dataset.py
```

Then run the machine learning project:

```bash
python house_price_prediction.py
```

## Dataset Source

Housing Price Prediction dataset:

https://github.com/MartinJessenne/Housing-Price-Prediction-Kaggle-Dataset

## Project Files

```text
HousePricePrediction/
│
├── download_dataset.py
├── house_price_prediction.py
├── requirements.txt
├── README.md
└── .gitignore
```

After running `download_dataset.py`, the following file will be added:

```text
Housing.csv
```

## Conclusion

This project shows how Linear Regression can be used to predict house prices. It also covers basic data preprocessing, feature scaling, train/test splitting and model evaluation.
