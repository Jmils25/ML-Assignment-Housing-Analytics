"""
House Price Prediction using Linear Regression
Author: ML Assignment Implementation
Data Source: Synthetic realistic data based on US housing market patterns (2020-2024)
             Prices modeled after metropolitan areas: Downtown, Suburb, Rural zones
             Square footage ranges: 800-4000 sq ft
             Price ranges: $150k-$900k based on location and size

Requirements:
- pandas
- numpy
- scikit-learn
- matplotlib
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate realistic housing dataset (150 records)
def generate_housing_data(n_samples=150):
    """
    Generate synthetic but realistic housing data
    
    Pricing assumptions:
    - Downtown: Base $300k + $250/sq ft
    - Suburb: Base $200k + $180/sq ft
    - Rural: Base $150k + $120/sq ft
    - Added random noise to simulate market variations
    """
    locations = ['Downtown', 'Suburb', 'Rural']
    location_base_prices = {
        'Downtown': 300000,
        'Suburb': 200000,
        'Rural': 150000
    }
    location_sqft_prices = {
        'Downtown': 250,
        'Suburb': 180,
        'Rural': 120
    }
    
    data = []
    
    for _ in range(n_samples):
        # Random location
        location = np.random.choice(locations, p=[0.3, 0.5, 0.2])  # More suburban houses
        
        # Square footage varies by location type
        if location == 'Downtown':
            square_footage = np.random.randint(800, 2500)
        elif location == 'Suburb':
            square_footage = np.random.randint(1200, 3500)
        else:  # Rural
            square_footage = np.random.randint(1000, 4000)
        
        # Calculate base price
        base_price = location_base_prices[location]
        sqft_price = location_sqft_prices[location]
        
        # Calculate price with some random variation (±10%)
        price = base_price + (square_footage * sqft_price)
        noise = np.random.normal(0, price * 0.10)  # 10% standard deviation
        price = max(price + noise, 50000)  # Ensure minimum price
        
        data.append({
            'square_footage': square_footage,
            'location': location,
            'price': round(price, 2)
        })
    
    return pd.DataFrame(data)

# Generate the dataset
print("=" * 70)
print("HOUSE PRICE PREDICTION - LINEAR REGRESSION MODEL")
print("=" * 70)
print("\n1. GENERATING REALISTIC HOUSING DATASET...")

df = generate_housing_data(150)

print(f"   Dataset size: {len(df)} records")
print(f"\n   Dataset preview:")
print(df.head(10))

print(f"\n   Dataset statistics:")
print(df.describe())

print(f"\n   Location distribution:")
print(df['location'].value_counts())

# Visualize the raw data
print("\n2. VISUALIZING RAW DATA...")

plt.figure(figsize=(12, 5))

# Scatter plot: Price vs Square Footage by Location
plt.subplot(1, 2, 1)
for location in df['location'].unique():
    location_data = df[df['location'] == location]
    plt.scatter(location_data['square_footage'], 
                location_data['price'], 
                label=location, 
                alpha=0.6)
plt.xlabel('Square Footage')
plt.ylabel('Price ($)')
plt.title('House Prices by Square Footage and Location')
plt.legend()
plt.grid(True, alpha=0.3)

# Box plot: Price distribution by Location
plt.subplot(1, 2, 2)
df.boxplot(column='price', by='location', ax=plt.gca())
plt.xlabel('Location')
plt.ylabel('Price ($)')
plt.title('Price Distribution by Location')
plt.suptitle('')  # Remove default title

plt.tight_layout()
plt.savefig('/home/claude/housing_data_visualization.png', dpi=300, bbox_inches='tight')
print("   Visualization saved as 'housing_data_visualization.png'")

# Prepare features and target
print("\n3. PREPARING DATA FOR TRAINING...")

X = df[['square_footage', 'location']]
y = df['price']

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Training set: {len(X_train)} samples")
print(f"   Testing set: {len(X_test)} samples")

# Create preprocessing pipeline
# OneHotEncoder for categorical 'location' variable
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['square_footage']),
        ('cat', OneHotEncoder(drop='first'), ['location'])  # Drop first to avoid multicollinearity
    ]
)

# Fit and transform the training data
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

print(f"   Features after encoding: {X_train_encoded.shape[1]}")
print(f"   Feature names: square_footage, location_Rural, location_Suburb")

# Train the Linear Regression model
print("\n4. TRAINING LINEAR REGRESSION MODEL...")

model = LinearRegression()
model.fit(X_train_encoded, y_train)

print("   Model training completed!")

# Make predictions on test set
y_pred = model.predict(X_test_encoded)

# Evaluate the model
print("\n5. MODEL EVALUATION:")

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"   Mean Squared Error (MSE): ${mse:,.2f}")
print(f"   Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"   Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"   R-squared (R²): {r2:.4f}")
print(f"   Model explains {r2*100:.2f}% of the variance in house prices")

# Display model coefficients
print("\n6. MODEL COEFFICIENTS (FEATURE IMPORTANCE):")

feature_names = ['square_footage', 'location_Rural', 'location_Suburb']
coefficients = model.coef_

print(f"   Intercept (Base price): ${model.intercept_:,.2f}")
print(f"\n   Coefficients:")
for name, coef in zip(feature_names, coefficients):
    if 'square_footage' in name:
        print(f"     • {name}: ${coef:.2f} per sq ft")
    else:
        print(f"     • {name}: ${coef:,.2f}")

print(f"\n   Interpretation:")
print(f"     - Each additional square foot increases price by ${coefficients[0]:.2f}")
print(f"     - Rural location adjusts price by ${coefficients[1]:,.2f} vs Downtown baseline")
print(f"     - Suburb location adjusts price by ${coefficients[2]:,.2f} vs Downtown baseline")

# Predict for a new house: 2000 sq ft in Downtown
print("\n7. PREDICTION FOR NEW HOUSE:")
print("   Specifications: 2000 sq ft, Downtown location")

new_house = pd.DataFrame({
    'square_footage': [2000],
    'location': ['Downtown']
})

new_house_encoded = preprocessor.transform(new_house)
predicted_price = model.predict(new_house_encoded)[0]

print(f"   Predicted Price: ${predicted_price:,.2f}")

# Additional predictions for comparison
print("\n   Comparative predictions (2000 sq ft):")
for loc in ['Downtown', 'Suburb', 'Rural']:
    test_house = pd.DataFrame({
        'square_footage': [2000],
        'location': [loc]
    })
    test_encoded = preprocessor.transform(test_house)
    test_pred = model.predict(test_encoded)[0]
    print(f"     • {loc}: ${test_pred:,.2f}")

# Visualize predictions vs actual
print("\n8. GENERATING PREDICTION VISUALIZATION...")

plt.figure(figsize=(12, 5))

# Actual vs Predicted scatter plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title(f'Actual vs Predicted Prices (R² = {r2:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)

# Residual plot
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Price ($)')
plt.ylabel('Residuals ($)')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/model_predictions_visualization.png', dpi=300, bbox_inches='tight')
print("   Visualization saved as 'model_predictions_visualization.png'")

# Save predictions to CSV
print("\n9. SAVING RESULTS...")

results_df = pd.DataFrame({
    'actual_price': y_test.values,
    'predicted_price': y_pred,
    'error': y_test.values - y_pred,
    'percent_error': ((y_test.values - y_pred) / y_test.values * 100)
})

results_df.to_csv('/home/claude/predictions_results.csv', index=False)
print("   Predictions saved to 'predictions_results.csv'")

# Save full dataset
df.to_csv('/home/claude/housing_dataset.csv', index=False)
print("   Full dataset saved to 'housing_dataset.csv'")

print("\n" + "=" * 70)
print("MODEL SUMMARY:")
print("=" * 70)
print(f"• Dataset: 150 houses across 3 locations")
print(f"• Model Accuracy (R²): {r2:.4f}")
print(f"• Average Prediction Error: ±${mae:,.2f}")
print(f"• Price per Square Foot: ${coefficients[0]:.2f}")
print(f"• Prediction for 2000 sq ft Downtown house: ${predicted_price:,.2f}")
print("=" * 70)
print("\nAnalysis complete! Check generated files for detailed results.")
