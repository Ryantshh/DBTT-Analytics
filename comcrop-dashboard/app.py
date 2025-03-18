from flask import Flask, render_template, jsonify
import os

app = Flask(__name__)

# Setup function to create required directories and files
def setup_app():
    """Set up required directories and files for the app."""
    # Create directories if they don't exist
    for directory in ['data', 'models', 'static/css', 'static/js', 'templates']:
        os.makedirs(os.path.join(app.root_path, directory), exist_ok=True)
    
    # Create water_data.csv if it doesn't exist
    csv_path = os.path.join(app.root_path, 'data', 'water_data.csv')
    if not os.path.exists(csv_path):
        csv_data = """Date,Day,Temperature,Humidity,Soil_Moisture,Rainfall,Water_Need
2025-03-17,17,29.7,76.0,49.5,0.0,10200.0
2025-03-18,18,30.0,77.0,50.0,0.0,12000.0
2025-03-19,19,30.2,77.5,50.5,1.0,10500.0
2025-03-20,20,30.5,78.0,51.0,2.0,12500.0
2025-03-21,21,30.3,77.8,50.8,1.5,11500.0
2025-03-22,22,30.0,77.0,50.0,0.0,11800.0
2025-03-23,23,29.8,76.5,49.5,0.5,11200.0"""
        with open(csv_path, 'w') as f:
            f.write(csv_data)

@app.route('/')
def dashboard():
    """Render the main dashboard page."""
    return render_template('dashboard.html')

@app.route('/api/water-prediction')
def water_prediction_api():
    """API endpoint for water prediction data."""
    # Import the module here to avoid circular imports
    from models.water_prediction import generate_water_prediction
    
    # Set path to CSV data
    csv_path = os.path.join(app.root_path, 'data', 'water_data.csv')
    
    # Make sure the file exists
    if not os.path.exists(csv_path):
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Create the CSV file with sample data
        csv_data = """Date,Day,Temperature,Humidity,Soil_Moisture,Rainfall,Water_Need
2025-03-17,17,29.7,76.0,49.5,0.0,10200.0
2025-03-18,18,30.0,77.0,50.0,0.0,12000.0
2025-03-19,19,30.2,77.5,50.5,1.0,10500.0
2025-03-20,20,30.5,78.0,51.0,2.0,12500.0
2025-03-21,21,30.3,77.8,50.8,1.5,11500.0
2025-03-22,22,30.0,77.0,50.0,0.0,11800.0
2025-03-23,23,29.8,76.5,49.5,0.5,11200.0"""
        with open(csv_path, 'w') as f:
            f.write(csv_data)
    
    # Generate water prediction data
    svg_code, metrics = generate_water_prediction(csv_path)
    
    # Return the data as JSON
    return jsonify({
        'svg': svg_code,
        'metrics': metrics
    })

if __name__ == '__main__':
    # Run setup before starting the app
    setup_app()
    app.run(debug=True)