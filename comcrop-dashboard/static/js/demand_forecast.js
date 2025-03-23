// Update the dashboard with demand forecast data
async function updateDemandForecast() {
    try {
        // Fetch the demand forecast data
        const response = await fetch('/api/demand-forecast');
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Update the demand forecast chart SVG
        const forecastChartContainer = document.querySelector('.demand-section .chart-container');
        if (forecastChartContainer && data.svg) {
            forecastChartContainer.innerHTML = data.svg;
        }
        
        // Update the product forecast chart SVG
        const productForecastContainer = document.querySelector('.product-forecast-container');
        if (productForecastContainer && data.product_forecast_svg) {
            productForecastContainer.innerHTML = data.product_forecast_svg;
        }
        
        // Update the metrics
        if (data.metrics) {
            updateForecastMetrics(data.metrics);
        }
        
        // Update timestamp
        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const forecastTimestamp = document.querySelector('.demand-section .timestamp');
        if (forecastTimestamp) {
            forecastTimestamp.textContent = `Last updated: ${timestamp}`;
        }
        
        console.log('Demand forecast updated successfully');
    } catch (error) {
        console.error('Error updating demand forecast:', error);
    }
}

// Update the demand forecast metrics in the dashboard
function updateForecastMetrics(metrics) {
    const demandSection = document.querySelector('.demand-section');
    if (!demandSection) return;
    
    const metricValues = demandSection.querySelectorAll('.metric-value');
    
    // Update Next Week Demand (first metric)
    if (metricValues[0] && metrics.next_week_demand) {
        metricValues[0].textContent = metrics.next_week_demand;
    }
    
    // Update Next Month Demand (second metric)
    if (metricValues[1] && metrics.next_month_demand) {
        metricValues[1].textContent = metrics.next_month_demand;
    }
    
    // Update Average Daily Demand (third metric)
    if (metricValues[2] && metrics.avg_daily_demand) {
        metricValues[2].textContent = metrics.avg_daily_demand;
    }
    
    // Update Peak Demand Day (fourth metric)
    if (metricValues[3] && metrics.peak_day) {
        metricValues[3].textContent = metrics.peak_day;
    }
    
    // Update demand change percentage with color coding
    const demandChangeElement = document.querySelector('.demand-change');
    if (demandChangeElement && metrics.demand_change) {
        demandChangeElement.textContent = metrics.demand_change;
        
        // Parse the percentage value
        const percentage = parseFloat(metrics.demand_change);
        
        // Apply color based on whether it's positive or negative
        if (percentage > 0) {
            demandChangeElement.classList.add('positive-change');
            demandChangeElement.classList.remove('negative-change');
        } else if (percentage < 0) {
            demandChangeElement.classList.add('negative-change');
            demandChangeElement.classList.remove('positive-change');
        } else {
            demandChangeElement.classList.remove('positive-change', 'negative-change');
        }
    }
    
    // Update top product section
    const topProductElement = document.querySelector('.top-product');
    if (topProductElement && metrics.top_product) {
        topProductElement.textContent = metrics.top_product;
    }
    
    // Update product breakdown SVG if present
    const productBreakdownContainer = document.querySelector('.product-breakdown-container');
    if (productBreakdownContainer && metrics.product_svg) {
        productBreakdownContainer.innerHTML = metrics.product_svg;
    }
}

// Initialize the demand forecast section
document.addEventListener('DOMContentLoaded', () => {
    // Check if we're on the demand forecast tab or if all sections are visible
    const demandSection = document.querySelector('.demand-section');
    if (!demandSection) return;
    
    console.log('Initializing demand forecast section...');
    
    // Initial update
    updateDemandForecast();
    
    // Set up refresh interval (every 5 minutes)
    setInterval(updateDemandForecast, 5 * 60 * 1000);
    
    // Setup tab switching functionality if tabs exist
    const tabs = document.querySelectorAll('.dashboard-tab');
    if (tabs.length > 0) {
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                // Remove active class from all tabs
                document.querySelectorAll('.dashboard-tab').forEach(t => {
                    t.classList.remove('active');
                });
                
                // Add active class to clicked tab
                tab.classList.add('active');
                
                // Hide all section content
                document.querySelectorAll('.dashboard-content').forEach(content => {
                    content.style.display = 'none';
                });
                
                // Show the content corresponding to the clicked tab
                const targetContentId = tab.getAttribute('data-target');
                const targetContent = document.getElementById(targetContentId);
                if (targetContent) {
                    targetContent.style.display = 'block';
                    
                    // If switching to demand forecast tab, refresh the data
                    if (targetContentId === 'demand-forecast-content') {
                        updateDemandForecast();
                    }
                }
            });
        });
    }
});