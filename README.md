# ComCrop Dashboard

A comprehensive analytics dashboard for ComCrop urban farm operations, featuring:

- Real-time environmental monitoring
- Water needs prediction
- Demand forecasting
- AI farming assistant

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Keys](#api-keys)
- [Running the Application](#running-the-application)
- [Dashboard Sections](#dashboard-sections)
- [Data Sources](#data-sources)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## Features

- **Farm Operations Dashboard**: Monitor environmental conditions, drone fleet status, water needs, disease risk, and nutrient levels
- **Demand Forecast Analytics**: View sales trends, product-specific forecasts, and planting recommendations
- **AI Farming Assistant**: Get real-time insights and answers to farming questions powered by Groq's language models

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/comcrop-dashboard.git
   cd comcrop-dashboard
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r comcrop-dashboard/requirements.txt
   ```

## Configuration

The dashboard requires configuration of environment variables for proper operation, particularly for the AI assistant functionality.

### Environment Variables

Create a `.env` file in the `comcrop-dashboard` directory with the following content:

```
GROQ_API_KEY=your_groq_api_key_here
```

## API Keys

### Groq API Key

The AI assistant uses the Groq API for processing queries. To obtain an API key:

1. Create an account at [Groq Console](https://console.groq.com/)
2. Navigate to API Keys section
3. Create a new API key
4. Copy the key and add it to your `.env` file as shown above

**Important Notes:**
- Never commit your `.env` file to version control
- Keep your API keys secure and do not share them
- The application will not function correctly without valid API keys
- If you're deploying to production, use environment variables instead of `.env` files

## Running the Application

1. Ensure you're in the project directory with the virtual environment activated
2. Run the Flask application:
   ```bash
   cd comcrop-dashboard
   python app.py
   ```
3. Open your browser and navigate to `http://127.0.0.1:5000/`

## Dashboard Sections

### Farm Operations

This tab displays:
- Current environmental conditions (temperature, humidity, soil moisture, etc.)
- Drone fleet management and scheduling
- Water needs prediction and irrigation scheduling
- Disease risk assessment visualization
- Nutrient deficiency monitoring

### Demand Forecast

This tab shows:
- Sales trend forecasts
- Product performance analysis
- Resource usage metrics
- Planting recommendations

### AI Bot

An interactive assistant that can:
- Answer questions about farm conditions
- Provide demand insights
- Suggest farming strategies
- Analyze trends in the data

## Data Sources

The dashboard uses several data sources:

- `water_data.csv`: Contains irrigation and water need records
- `historical_data_random.csv`: Contains historical sales data
- `historical_data_random11.csv`: Additional historical sales data

The application will automatically create sample data files if they don't exist.

## Development

### Project Structure

```
comcrop-dashboard/
├── app.py                   # Main Flask application
├── data/                    # Data files
│   ├── historical_data_random.csv
│   ├── historical_data_random11.csv
│   └── water_data.csv
├── models/                  # Analysis and prediction models
│   ├── aibot.py             # AI assistant implementation
│   ├── demand_forecast.py   # Sales forecasting logic
│   └── water_prediction.py  # Water needs prediction
├── requirements.txt         # Python dependencies
├── static/                  # Static assets
│   ├── css/
│   │   └── dashboard.css
│   └── js/
│       ├── aibot.js
│       ├── dashboard.js
│       └── demand_forecast.js
└── templates/               # HTML templates
    └── dashboard.html
```

### Adding New Features

When developing new features:

1. Add new data sources to the `data/` directory
2. Create new model files in the `models/` directory
3. Add new endpoints to `app.py`
4. Update the UI in `templates/dashboard.html` and add styling in `static/css/dashboard.css`
5. Add client-side logic in JavaScript files in `static/js/`

## Troubleshooting

### Common Issues

**API Key Issues:**
- If you encounter errors with the AI assistant, verify your Groq API key is correctly set in the `.env` file
- Check the console logs for API-related errors

**Data Visualization Issues:**
- If charts are not displaying, check browser console for JavaScript errors
- Verify data files exist in the `data/` directory

**Python Errors:**
- Make sure all dependencies are installed
- Check Python logs for tracebacks
- Ensure you're using Python 3.8+

### Getting Help

If you encounter issues:
1. Check the logs for error messages
2. Consult the documentation
3. Create an issue on the GitHub repository with detailed information about the problem

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Flask](https://flask.palletsprojects.com/)
- [Groq API](https://groq.com/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
