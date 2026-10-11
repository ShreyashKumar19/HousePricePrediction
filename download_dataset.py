import urllib.request

url = "https://raw.githubusercontent.com/MartinJessenne/Housing-Price-Prediction-Kaggle-Dataset/main/Housing.csv"

urllib.request.urlretrieve(url, "Housing.csv")

print("Dataset downloaded successfully.")