# energy-insight

To create an AI-powered dashboard for real-time energy consumption monitoring and optimization, you can design a Python program that incorporates data collection, data processing, machine learning for predictive insights, and a web interface for visualization. Below is a simplified version to get you started. For a real-world application, you'll need to expand upon each section, especially the integration with IoT devices, real-time data streams, and sophisticated ML models.

Here's a basic outline for such a project with comments and error handling:

```python
import random
import time
import logging
from flask import Flask, jsonify, render_template
import numpy as np
from sklearn.linear_model import LinearRegression

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)s: %(message)s')

app = Flask(__name__)

# Simulate real-time energy consumption data
def simulate_energy_data():
    try:
        current_time = time.localtime()
        hour = current_time.tm_hour
        # Simulate energy consumption as a random number adjusted for time of day
        base = 50
        if 6 <= hour < 18:  # Daytime hours, higher consumption
            consumption = base + random.uniform(20, 50)
        else:  # Nighttime hours, lower consumption
            consumption = base + random.uniform(5, 20)
        usage_data.append(consumption)
        logging.info(f"Energy data simulated: {consumption} kWh")
        return consumption
    except Exception as e:
        logging.error(f"Error in simulating energy data: {e}")
        return None

# Initialize a list to store energy usage data
usage_data = []

# Train a simple machine learning model
def train_model():
    try:
        # Generate some synthetic training data
        X = np.arange(24).reshape(-1, 1)  # Hours of the day
        y = [simulate_energy_data() for _ in range(24)]  # Simulate data for a day

        # Train a linear regression model
        model = LinearRegression()
        model.fit(X, y)
        logging.info("Machine learning model trained successfully")
        return model
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        return None

# A simple prediction function
def predict_energy_usage(model, input_hour):
    try:
        if not model:
            raise ValueError("Model is not available")
        prediction = model.predict(np.array([[input_hour]]))[0]
        logging.info(f"Predicted energy usage for hour {input_hour}: {prediction} kWh")
        return prediction
    except Exception as e:
        logging.error(f"Error in predicting energy usage: {e}")
        return None

# Initialize the machine learning model
model = train_model()

# Define routes for the Flask app
@app.route('/')
def dashboard():
    try:
        # Use Flask to serve an HTML page (assuming 'index.html' exists in a 'templates' folder)
        return render_template('index.html', usage_data=usage_data)
    except Exception as e:
        logging.error(f"Error in loading dashboard page: {e}")
        return "Error loading dashboard"

@app.route('/data')
def get_data():
    try:
        # Provide the latest energy consumption data
        return jsonify(usage_data)
    except Exception as e:
        logging.error(f"Error in retrieving data: {e}")
        return "Error retrieving data"

@app.route('/predict/<int:hour>')
def get_prediction(hour):
    try:
        prediction = predict_energy_usage(model, hour)
        if prediction is None:
            raise ValueError("Prediction failed")
        return jsonify(predicted_usage=prediction)
    except Exception as e:
        logging.error(f"Error in making prediction: {e}")
        return "Error predicting energy usage"

if __name__ == '__main__':
    # Populate initial data
    for _ in range(24):
        simulate_energy_data()

    # Run the Flask app
    app.run(debug=True)
```

### Explanation:

1. **Simulate Data Collection**: This section simulates real-time energy consumption data. In a real application, this could come from smart meters or IoT devices.

2. **Machine Learning Model**: A simple linear regression model is used here for prediction based on the time of day. You can replace this with a more sophisticated model depending on your needs.

3. **Flask Web Application**: Flask is used to create a web server that can render a dashboard and provide API endpoints to retrieve data and predictions.

4. **Error Handling**: Included in each function to log errors and ensure the application remains robust.

5. **Logging**: Important for debugging and monitoring the application's performance.

You will need to set up a proper HTML page (`index.html`) in the `templates` directory and augment the program with real data integration, real-time streaming capabilities, and more sophisticated models to build a complete system.