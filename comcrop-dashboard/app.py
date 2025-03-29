from turtle import pd
from venv import logger
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask import jsonify, request
from models.aibot import AIBot
from dotenv import load_dotenv
load_dotenv()
import logging
import traceback
import asyncio
import os

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  
aibot = AIBot()

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

@app.route('/api/demand-forecast')
def demand_forecast_api():
    """API endpoint for demand forecast data."""
    # Import the module here to avoid circular imports
    from models.demand_forecast import generate_forecast_data, get_forecast_by_product
    
    # Set path to CSV data
    csv_path = os.path.join(app.root_path, 'data', 'historical_data_random.csv')
    
    # Generate demand forecast data
    svg_code, metrics = generate_forecast_data(csv_path)
    
    # Get product-specific forecasts
    product_forecast_svg = get_forecast_by_product(csv_path)
    
    # Return the data as JSON
    return jsonify({
        'svg': svg_code,
        'metrics': metrics,
        'product_forecast_svg': product_forecast_svg
    })
    
# Update your API endpoint:
@app.route('/api/ai-query', methods=['POST'])
def handle_ai_query():
    """Handle AI bot requests"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Missing query parameter"}), 400
            
        response = aibot.query(data['query'])
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"API Error: {traceback.format_exc()}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e),
            "success": False
        }), 500


def get_latest_water_data():
    """Helper to get current water prediction data"""
    csv_path = os.path.join(app.root_path, 'data', 'water_data.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df.iloc[-1].to_dict()
    return {}

def get_latest_demand_data():
    """Helper to get current demand forecast data"""
    csv_path = os.path.join(app.root_path, 'data', 'historical_data_random.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return {
            "latest_sales": df.iloc[-1].to_dict(),
            "top_products": df.groupby('Name')['Quantity Sold'].sum().nlargest(3).to_dict()
        }
    return {}



if __name__ == '__main__':
    # Run setup before starting the app
    setup_app()
    app.run(debug=True)