// Update the dashboard with fresh water prediction data
async function updateWaterPrediction() {
    try {
        // Fetch the water prediction data
        const response = await fetch('/api/water-prediction');
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Update the water chart SVG in the water section
        const waterChartContainer = document.querySelector('.water-section .chart-container');
        if (waterChartContainer && data.svg) {
            waterChartContainer.innerHTML = data.svg;
        }
        
        // Update the metrics
        if (data.metrics) {
            updateWaterMetrics(data.metrics);
        }
        
        // Update timestamp
        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const waterTimestamp = document.querySelector('.water-section .timestamp');
        if (waterTimestamp) {
            waterTimestamp.textContent = `Last updated: ${timestamp}`;
        }
        
        console.log('Water prediction updated successfully');
    } catch (error) {
        console.error('Error updating water prediction:', error);
    }
}

// Update the water metrics in the dashboard
function updateWaterMetrics(metrics) {
    const waterSection = document.querySelector('.water-section');
    if (!waterSection) return;
    
    const metricValues = waterSection.querySelectorAll('.metric-value');
    
    // Update Today's Need (first metric)
    if (metricValues[0] && metrics.today_need) {
        metricValues[0].textContent = metrics.today_need;
    }
    
    // Update Next Irrigation (second metric)
    if (metricValues[1] && metrics.next_irrigation) {
        metricValues[1].textContent = metrics.next_irrigation;
    }
    
    // Update 7-Day Forecast (third metric)
    if (metricValues[2] && metrics.forecast_total) {
        metricValues[2].textContent = metrics.forecast_total;
    }
    
    // Update Soil Moisture (fourth metric)
    if (metricValues[3] && metrics.soil_moisture) {
        metricValues[3].textContent = metrics.soil_moisture;
    }
}

// Add event handlers for buttons
function setupButtonHandlers() {
    // Export Data button
    const exportButton = document.querySelector('.header-actions button:nth-child(1)');
    if (exportButton) {
        exportButton.addEventListener('click', () => {
            alert('Exporting data...');
            // Implement export functionality
        });
    }
    
    // Settings button
    const settingsButton = document.querySelector('.header-actions button:nth-child(2)');
    if (settingsButton) {
        settingsButton.addEventListener('click', () => {
            alert('Settings panel would open here');
            // Implement settings functionality
        });
    }
    
    // Help button
    const helpButton = document.querySelector('.header-actions button:nth-child(3)');
    if (helpButton) {
        helpButton.addEventListener('click', () => {
            alert('Help documentation would open here');
            // Implement help functionality
        });
    }
}

// Initialize the dashboard
document.addEventListener('DOMContentLoaded', () => {
    console.log('Dashboard initializing...');
    
    // Set up button handlers
    setupButtonHandlers();
    
    // Initial update for water prediction
    updateWaterPrediction();
    
    // Set up refresh interval (every 5 minutes)
    setInterval(updateWaterPrediction, 5 * 60 * 1000);
});