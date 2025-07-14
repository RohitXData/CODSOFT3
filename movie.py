# movie_rating_simple.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load local CSV
data = pd.read_csv('movies.csv')

print("Dataset loaded.")
print(data)

# Split features & target
X = data[['budget', 'runtime', 'year']]
y = data['imdb_rating']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)

print(f"\nR2 Score: {score:.2f}")
print(f"Predicted Ratings: {y_pred}")