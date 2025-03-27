import os
import time
from groq import Groq
import logging

import pandas as pd

logger = logging.getLogger(__name__)

class AIBot:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        # Updated to current models (June 2024)
        self.model = "mixtral-8x7b-32768"  # Fallback
        self.available_models = [
            "llama3-70b-8192",  # New recommended model
            "llama3-8b-8192",   # Faster alternative
            "mixtral-8x7b-32768" # Fallback if others fail
        ]
        
        self.system_prompt = """You are Comcrop's farming assistant AI. Your responses MUST:
            1. Reference current water and demand data when relevant
            2. Explain trends using the provided numbers
            3. Suggest actions based on the data
            4. Keep responses concise but informative

            Current Data will be provided with each query."""
        
        self.conversation_history = []
        self.current_model = self._get_working_model()
        self.data_context = {
            "water_data": None,
            "demand_data": None
        }
        self._refresh_data()
    
    def _refresh_data(self):
        """Load the latest data from CSV files"""
        try:
            # Water data
            water_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'water_data.csv')
            if os.path.exists(water_path):
                water_df = pd.read_csv(water_path)
                self.data_context["water_data"] = water_df.iloc[-1].to_dict()
            
            # Demand data
            demand_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'historical_data_random.csv')
            if os.path.exists(demand_path):
                demand_df = pd.read_csv(demand_path)
                self.data_context["demand_data"] = {
                    "latest": demand_df.iloc[-1].to_dict(),
                    "top_products": demand_df.groupby('Name')['Quantity Sold'].sum().nlargest(3).to_dict()
                }
        except Exception as e:
            logger.error(f"Data refresh failed: {str(e)}")

    def _format_data_context(self):
        """Create a natural language summary of the data"""
        context = []
        
        # Water data summary
        if self.data_context["water_data"]:
            water = self.data_context["water_data"]
            context.append(
                f"Current water conditions:\n"
                f"- Temperature: {water.get('Temperature', 'N/A')}°C\n"
                f"- Soil Moisture: {water.get('Soil_Moisture', 'N/A')}%\n"
                f"- Water Need: {water.get('Water_Need', 'N/A')}L"
            )
        
        # Demand data summary
        if self.data_context["demand_data"]:
            demand = self.data_context["demand_data"]
            context.append(
                f"Latest demand insights:\n"
                f"- Recent sales: {demand['latest'].get('Quantity Sold', 'N/A')} units\n"
                f"- Top products: {', '.join(demand['top_products'].keys())}"
            )
        
        return "\n\n".join(context) if context else "No data available"

    def _get_working_model(self):
        """Find the first available working model"""
        for model in self.available_models:
            try:
                # Test with a simple query
                self.client.chat.completions.create(
                    messages=[{"role": "user", "content": "test"}],
                    model=model,
                    max_tokens=1
                )
                logger.info(f"Using model: {model}")
                return model
            except Exception as e:
                logger.warning(f"Model {model} not available: {str(e)}")
        raise RuntimeError("No available models found")

    def query(self, user_input):
        """Process user query with automatic model fallback"""
        self._refresh_data()  # Get fresh data before each query
        

        start_time = time.time()  # Add import time at top of file

        try:
            messages = [
                {
                    "role": "system",
                    "content": f"{self.system_prompt}\n\nCurrent Data:\n{self._format_data_context()}"
                },
                *self.conversation_history[-6:],
                {"role": "user", "content": user_input}
            ]
            
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.current_model,
                temperature=0.7,
                max_tokens=1024
            )
            
            response_time = time.time() - start_time
            bot_response = response.choices[0].message.content
            self._update_conversation(user_input, bot_response)
            
            return {
                "response": bot_response,
                "model": self.current_model,
                "response_time": f"{response_time:.2f}s",  # Formatted time
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            # Try with next available model
            try:
                self.current_model = self._get_working_model()
                return self.query(user_input)  # Retry
            except Exception as fallback_error:
                return {
                    "response": "Our AI systems are currently unavailable. Please try again later.",
                    "error": str(fallback_error),
                    "success": False
                }

    def _update_conversation(self, user_input, bot_response):
        """Maintain conversation history"""
        self.conversation_history.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": bot_response}
        ])