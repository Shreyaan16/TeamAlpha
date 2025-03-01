import streamlit as st
import requests
import tensorflow as tf
import json
import os
import base64
import tempfile
import pandas as pd
import numpy as np
import urllib.request
import plotly.express as px
import re
import joblib
import warnings
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import logging
from tensorflow.keras.preprocessing import image
import folium
from streamlit_folium import folium_static


# Set page configuration (MUST be the first Streamlit command)
st.set_page_config(
    page_title="Weather, Soil & Crop Analysis App",
    page_icon="🌦️",
    layout="wide"
)

# Load API keys from .env file
load_dotenv()

# Initialize session state for API keys
if "weather_apikey" not in st.session_state:
    st.session_state["weather_apikey"] = os.getenv("WEATHER_API_KEY", "")

if "gemini_apikey" not in st.session_state:
    st.session_state["gemini_apikey"] = os.getenv("GEMINI_API_KEY", "")

def save_api_keys():
    """Save API keys to session state and update environment variables."""
    os.environ["WEATHER_API_KEY"] = st.session_state["weather_apikey"]
    os.environ["GEMINI_API_KEY"] = st.session_state["gemini_apikey"]
    st.success("✅ API Keys updated successfully!")

# Sidebar UI for API keys
with st.sidebar:
    st.markdown("### 🔑 Enter API Keys")
    st.markdown("---")  # Adds a horizontal divider

    st.text_input("🌤️ Weather API Key", key="weather_apikey", type="password")
    st.text_input("💎 Gemini API Key", key="gemini_apikey", type="password")

    st.markdown("<br>", unsafe_allow_html=True)  # Adds spacing
    st.button("💾 Save API Keys", on_click=save_api_keys, use_container_width=True)

# Use the updated API keys
API_KEY = st.session_state["gemini_apikey"]
WEATHER_API_KEY = st.session_state["weather_apikey"]

# Suppress warnings
warnings.filterwarnings("ignore")


# Load API keys from .env file
weather_apikey = os.getenv("WEATHER_API_KEY")
gemini_apikey = os.getenv("GEMINI_API_KEY")

# Load trained model and scaler
model = joblib.load("recommender.pkl")
scaler = joblib.load("recommender_scaled.pkl")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Session state initialization for data sharing between tabs
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None
if 'soil_data' not in st.session_state:
    st.session_state.soil_data = None
if 'city' not in st.session_state:
    st.session_state.city = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'recommended_crop' not in st.session_state:
    st.session_state.recommended_crop = None

# Create tabs for the application
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌦️ Weather Forecaster", 
    "🧑‍🌾 Soil Health Card Analyzer", 
    "🌱 Crop Recommendation", 
    "💬 Agricultural Advisor", 
    "🐛 Pest & Disease Remedies", 
    "📍 Geofencing"
])


# Function to analyze the image using Gemini API
def analyze_image(api_key, image_path, prompt):
    base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    
    encoded_image = encode_image(image_path)
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": encoded_image
                        }
                    }
                ]
            }
        ],
        "generation_config": {
            "temperature": 0.4,
            "top_p": 1,
            "top_k": 32,
            "max_output_tokens": 4096
        }
    }
    
    response = requests.post(
        f"{base_url}?key={api_key}",
        headers={"Content-Type": "application/json"},
        json=payload
    )
    
    return response.json()

# Function to send a query to the Gemini API for chat
def query_gemini(api_key, prompt):
    base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generation_config": {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 4096
        }
    }
    
    response = requests.post(
        f"{base_url}?key={api_key}",
        headers={"Content-Type": "application/json"},
        json=payload
    )
    
    return response.json()

# Function to encode image to Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Base URL for weather API
BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

# Cache function to improve performance
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_weather_data(city):
    """Fetch weather data from Visual Crossing API."""
    if not weather_apikey:
        st.error("❌ Error: Weather API key is missing! Set WEATHER_API_KEY in your .env file.")
        return None

    url = f"{BASE_URL}/{city}?unitGroup=us&key={weather_apikey}&contentType=json"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"❌ Error fetching weather data: HTTP {response.status_code}")
            st.write(response.text)
            return None
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
        return None

def extract_specific_weather_data(weather_data):
    """Extract key weather details from API response."""
    current = weather_data.get("currentConditions", {})

    specific_data = {
        "city": weather_data.get("resolvedAddress", "Unknown location"),
        "description": weather_data.get("description", "No description available"),
        "current": {
            "temp": current.get("temp"),
            "temp_c": (current.get("temp") - 32) * 5/9,  # Convert F to C for the model
            "temp_k": ((current.get("temp") - 32) * 5/9) + 273.15,  # Convert F to K
            "feelslike": current.get("feelslike"),
            "humidity": current.get("humidity"),
            "precipitation": current.get("precip", 0),
            "precipprob": current.get("precipprob", 0),
            "windspeed": current.get("windspeed"),
            "pressure": current.get("pressure"),
            "solarradiation": current.get("solarradiation"),
            "solarenergy": current.get("solarenergy"),
            "condition": current.get("conditions"),
            "description": current.get("description", weather_data.get("description", "No description available"))
        },
        "forecast": []
    }

    for day in weather_data.get("days", []):
        day_data = {
            "date": day.get("datetime"),
            "temp": day.get("temp"),
            "temp_c": (day.get("temp") - 32) * 5/9,  # Convert F to C for the model
            "feelslike": day.get("feelslike"),
            "humidity": day.get("humidity"),
            "precipitation": day.get("precip", 0),
            "precipprob": day.get("precipprob", 0),
            "windspeed": day.get("windspeed"),
            "pressure": day.get("pressure"),
            "solarradiation": day.get("solarradiation"),
            "solarenergy": day.get("solarenergy"),
            "condition": day.get("conditions"),
            "description": day.get("description", "No description available")
        }
        specific_data["forecast"].append(day_data)

    return specific_data

# Function to create forecast dataframe
def create_forecast_df(forecast_data):
    df = pd.DataFrame(forecast_data)
    # Convert date string to datetime
    df['date'] = pd.to_datetime(df['date'])
    # Format date for display
    df['formatted_date'] = df['date'].dt.strftime('%a, %b %d')
    return df

# Function to display weather icon based on condition
def get_weather_emoji(condition):
    condition = condition.lower() if condition else ""
    if "rain" in condition:
        return "🌧️"
    elif "snow" in condition:
        return "❄️"
    elif "cloud" in condition:
        return "☁️"
    elif "clear" in condition or "sun" in condition:
        return "☀️"
    elif "storm" in condition or "thunder" in condition:
        return "⛈️"
    elif "fog" in condition or "mist" in condition:
        return "🌫️"
    elif "wind" in condition:
        return "💨"
    else:
        return "🌤️"

# Weather Forecaster Page
with tab1:
    st.title("🌦️ Weather Forecast App")

    # City input
    city = st.text_input("Enter a city name:", "")
    st.session_state.city = city  # Save to session state for chatbot

    # Button to fetch weather data
    if st.button("Get Weather Forecast", key="weather_button") or city:
        with st.spinner("Fetching weather data..."):
            weather_data = fetch_weather_data(city)
            
            if weather_data:
                specific_data = extract_specific_weather_data(weather_data)
                st.session_state.weather_data = specific_data  # Save to session state for chatbot
                
                # Display current weather
                st.header(f"Current Weather in {specific_data['city']} {get_weather_emoji(specific_data['current']['condition'])}")
                
                # Create columns for current weather details
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Temperature", f"{specific_data['current']['temp']}°F")
                    st.metric("Feels Like", f"{specific_data['current']['feelslike']}°F")
                
                with col2:
                    st.metric("Humidity", f"{specific_data['current']['humidity']}%")
                    st.metric("Precipitation", f"{specific_data['current']['precipitation']} in")
                
                with col3:
                    st.metric("Wind Speed", f"{specific_data['current']['windspeed']} mph")
                    st.metric("Pressure", f"{specific_data['current']['pressure']} mb")
                    
                with col4:
                    st.metric("Condition", specific_data['current']['condition'])
                    st.metric("Precipitation Probability", f"{specific_data['current']['precipprob']}%")
                
                st.write(f"**Description:** {specific_data['description']}")
                
                # Display forecast
                st.header("14-Day Forecast")
                
                # Create a dataframe for the forecast
                forecast_df = create_forecast_df(specific_data['forecast'])
                
                # Temperature chart
                st.subheader("Temperature Forecast")
                fig = px.line(
                    forecast_df, 
                    x='formatted_date', 
                    y=['temp', 'feelslike'], 
                    title="Temperature Forecast (°F)",
                    labels={'value': 'Temperature (°F)', 'formatted_date': 'Date', 'variable': 'Measurement'},
                    color_discrete_map={'temp': '#FF9914', 'feelslike': '#FFBF69'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Precipitation chart
                st.subheader("Precipitation Forecast")
                fig = px.bar(
                    forecast_df, 
                    x='formatted_date', 
                    y='precipprob', 
                    title="Precipitation Probability (%)",
                    labels={'precipprob': 'Probability (%)', 'formatted_date': 'Date'},
                    color='precipprob',
                    color_continuous_scale='blues'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display detailed forecast as a table
                st.subheader("Detailed Forecast")
                
                # Create a more readable table
                display_df = forecast_df[['formatted_date', 'temp', 'feelslike', 'humidity', 'precipitation', 'precipprob', 'windspeed', 'condition']].copy()
                display_df.columns = ['Date', 'Temp (°F)', 'Feels Like (°F)', 'Humidity (%)', 'Precip (in)', 'Precip Prob (%)', 'Wind (mph)', 'Condition']
                
                # Add weather emoji to condition
                display_df['Condition'] = display_df['Condition'].apply(lambda x: f"{get_weather_emoji(x)} {x}")
                
                st.dataframe(display_df, use_container_width=True)
                
                # Option to download data
                st.download_button(
                    label="Download Weather Data (JSON)",
                    data=json.dumps(specific_data, indent=2),
                    file_name=f"{city}_weather_data.json",
                    mime="application/json"
                )
                
                # Add a map showing the location
                st.subheader("Location")
                try:
                    latitude = weather_data.get("latitude", 0)
                    longitude = weather_data.get("longitude", 0)
                    
                    if latitude and longitude:
                        df_map = pd.DataFrame({
                            'lat': [latitude],
                            'lon': [longitude],
                            'name': [specific_data['city']]
                        })
                        st.map(df_map)
                    else:
                        st.info("Location coordinates not available for mapping.")
                except Exception as e:
                    st.warning(f"Could not display map: {e}")
            else:
                st.error("Failed to fetch weather data. Please check the city name and try again.")

    # Footer
    st.markdown("---")
   

# Soil Health Card Reader Page
with tab2:
    st.title("🧑‍🌾 Soil Health Card Analyzer")
    st.write("Upload a health card image to extract soil properties.")

    # Upload image
    uploaded_file = st.file_uploader("Upload Health Card Image", type=["png", "jpg", "jpeg"], key="soil_uploader")

    if uploaded_file:
        # Save uploaded file
        image_path = "uploaded_health_card.jpg"
        image2 = Image.open(uploaded_file)
        
        # Fix RGBA to RGB issue
        if image2.mode == "RGBA":
            image2 = image2.convert("RGB")
        
        image2.save(image_path, "JPEG")  # Save as JPEG
        
        # Display uploaded image
        st.image(image2, use_container_width=True)
        
        # Check for API Key
        if not gemini_apikey:
            st.error("❌ Gemini API key missing! Set GEMINI_API_KEY in your .env file.")
            st.stop()
        
        # JSON Prompt for Gemini API
        json_prompt = """Analyze this image and provide ONLY a JSON object with these values and absolutely nothing else:
        {
          "pH": "value",
          "N": "value",
          "P": "value",
          "K": "value",
          "S": "value",
          "Zn": "value",
          "B": "value",
          "Fe": "value",
          "Mn": "value",
          "Cu": "value",
          "OC": "value",
          "Soil_type": "value",
          "Farm_size(acres)": "value",
          "Longitude": "value",
          "Latitude": "value",
          "Irrigation_status": "string",
          "EC": "value"
        }
        For N, P, K, pH values - ensure they are numeric values (remove any units and convert to numbers).
        """
        
        # Perform image analysis
        analyze_button = st.button("Analyze Health Card", key="analyze_button")
        
        if analyze_button:
            with st.spinner("Analyzing Image... ⏳"):
                analysis_result = analyze_image(gemini_apikey, image_path, json_prompt)

            try:
                # Extract JSON from response
                text_content = analysis_result["candidates"][0]["content"]["parts"][0]["text"]
                json_pattern = r"\{[\s\S]*\}"
                json_match = re.search(json_pattern, text_content)
                
                if json_match:
                    json_str = json_match.group(0)
                    parsed_json = json.loads(json_str)
                    
                    # Convert string values to numeric where needed
                    for key in ['N', 'P', 'K', 'pH']:
                        if key in parsed_json:
                            try:
                                parsed_json[key] = float(parsed_json[key].replace('kg/ha', '').strip())
                            except (ValueError, AttributeError):
                                # Keep as is if conversion fails
                                pass
                    
                    # Save extracted data to session state for chatbot and crop recommendation
                    st.session_state.soil_data = parsed_json
                    
                    # Save extracted data
                    with open("soil_analysis.json", "w") as f:
                        json.dump(parsed_json, f, indent=2)
                    
                    # Create two columns for display
                    col1, col2 = st.columns(2)
                    
                    # Display extracted values in a more organized way
                    st.subheader("🔍 Extracted Soil Data")
                    
                    # Create a dataframe for better visualization
                    soil_data = []
                    for key, value in parsed_json.items():
                        soil_data.append({"Parameter": key, "Value": value})
                    
                    soil_df = pd.DataFrame(soil_data)
                    st.dataframe(soil_df, use_container_width=True)
                    
                    # Option to download data
                    st.download_button(
                        label="Download Soil Analysis (JSON)",
                        data=json.dumps(parsed_json, indent=2),
                        file_name="soil_analysis_data.json",
                        mime="application/json"
                    )
                    
                else:
                    st.error("❌ No valid JSON found in API response.")
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
                st.json(analysis_result)  # Show raw response for debugging

    # Footer
    st.markdown("---")

# Crop Recommendation Tab
# Crop Recommendation Tab
with tab3:
    st.title("🌱 Crop Recommendation System")
    st.write("Get crop recommendations based on soil and weather data.")

    # Check if we have both weather and soil data
    if st.session_state.weather_data and st.session_state.soil_data:
        st.success("✅ Using data from Weather Forecast and Soil Analysis")
        
        # Get soil parameters from stored data
        try:
            # Extract and convert to float if needed, with default value of 0 for missing data
            N = float(st.session_state.soil_data.get('N', 0)) if st.session_state.soil_data.get('N') not in [None, ''] else 0
            P = float(st.session_state.soil_data.get('P', 0)) if st.session_state.soil_data.get('P') not in [None, ''] else 0
            K = float(st.session_state.soil_data.get('K', 0)) if st.session_state.soil_data.get('K') not in [None, ''] else 0
            ph = float(st.session_state.soil_data.get('pH', 7.0)) if st.session_state.soil_data.get('pH') not in [None, ''] else 7.0
            
            # Get weather data and convert units, with default values if missing
            temp_celsius = st.session_state.weather_data['current'].get('temp_c', 25.0) if st.session_state.weather_data['current'].get('temp_c') is not None else 25.0
            humidity = st.session_state.weather_data['current'].get('humidity', 60.0) if st.session_state.weather_data['current'].get('humidity') is not None else 60.0
            
            # Convert precipitation from inches to mm, with default value if missing
            rainfall_mm = st.session_state.weather_data['current'].get('precipitation', 0) * 25.4 if st.session_state.weather_data['current'].get('precipitation') is not None else 0
            
            # Display the parameters being used
            st.subheader("Parameters for crop recommendation:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Soil Parameters:**")
                st.write(f"Nitrogen (N): {N:.2f}")
                st.write(f"Phosphorus (P): {P:.2f}")
                st.write(f"Potassium (K): {K:.2f}")
                st.write(f"pH: {ph:.2f}")
            
            with col2:
                st.write(f"**Weather Parameters:**")
                st.write(f"Temperature: {temp_celsius:.2f}°C")
                st.write(f"Humidity: {humidity:.2f}%")
                st.write(f"Rainfall: {rainfall_mm:.2f} mm")
            
            # Button to get recommendation
            if st.button("Get Crop Recommendation"):
                with st.spinner("Analyzing data..."):
                    # Prepare input for model - using only the required parameters
                    # Model expects [N, P, K, temp, humidity, ph, rainfall]
                    user_input = np.array([[N, P, K, temp_celsius, humidity, ph, rainfall_mm]])
                    
                    # Apply scaling
                    scaled_input = scaler.transform(user_input)
                    
                    # Make prediction
                    prediction = model.predict(scaled_input)
                    
                    # Store the result in session state for chatbot
                    st.session_state.recommended_crop = prediction[0]
                    
                    # Display result with a nice UI
                    st.success(f"🌾 Based on your soil and weather conditions, the recommended crop is:")
                    st.markdown(f"<h1 style='text-align: center; color: #2e7d32;'>{prediction[0]}</h1>", unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"⚠️ Error making prediction: {e}")
            st.warning("Some parameters might be missing or in incorrect format. Please check your soil health card data.")
            
    else:
        st.warning("⚠️ Complete both Weather Forecast and Soil Health Card Analysis tabs first")
        
        # Manual input option
        st.subheader("Or enter parameters manually:")
        
        N = st.slider("Nitrogen Content (N)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
        P = st.slider("Phosphorus Content (P)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
        K = st.slider("Potassium Content (K)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
        temp = st.slider("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
        humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
        ph = st.slider("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        rainfall = st.slider("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0, step=1.0)
        
        if st.button("Predict Crop"):
            try:
                # Normalize input
                user_input = np.array([[N, P, K, temp, humidity, ph, rainfall]])
                scaled_input = scaler.transform(user_input)
                
                # Make prediction
                prediction = model.predict(scaled_input)
                
                # Store result for chatbot
                st.session_state.recommended_crop = prediction[0]
                
                # Display result
                st.success(f"🌾 Based on your input, the recommended crop is:")
                st.markdown(f"<h1 style='text-align: center; color: #2e7d32;'>{prediction[0]}</h1>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ Error making prediction: {e}")
                st.warning("There was an issue processing your input. Please try different values.")
    
    # Footer
    st.markdown("---")

# Agricultural Advisor Chatbot Page
with tab4:
    st.title("💬 Agricultural Advisor")
    
    # Introduction text
    st.write("Ask questions about weather conditions, soil health, and crop recommendations to get tailored agricultural advice.")
    
    # Status of data
    with st.expander("Data Status"):
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.session_state.weather_data:
                st.success(f"✅ Weather data available for {st.session_state.weather_data['city']}")
            else:
                st.warning("⚠️ No weather data available.")
        
        with col2:
            if st.session_state.soil_data:
                st.success("✅ Soil health data available")
            else:
                st.warning("⚠️ No soil health data available.")
        
        with col3:
            if st.session_state.recommended_crop:
                st.success(f"✅ Recommended crop: {st.session_state.recommended_crop}")
            else:
                st.warning("⚠️ No crop recommendation yet.")
    
    # Chat interface
    st.subheader("Chat with Agricultural Advisor")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"<div style='background-color: #e6f7ff; color: #333333; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background-color: #f0f0f0; color: #333333; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><strong>Advisor:</strong> {message['content']}</div>", unsafe_allow_html=True)
    
    # Initialize the input value key in session state if it doesn't exist
    if 'input_value' not in st.session_state:
        st.session_state.input_value = ""
    
    # Function to clear input after sending
    def clear_and_send():
        if st.session_state.input_value:
            # Store the current input value
            current_input = st.session_state.input_value
            
            # Clear the input value
            st.session_state.input_value = ""
            
            # Process the user message
            process_user_message(current_input)
    
    # Function to process user message
    def process_user_message(user_input):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Check for API Key
        if not gemini_apikey:
            st.session_state.chat_history.append({"role": "assistant", "content": "⚠️ Gemini API key is missing. Please set GEMINI_API_KEY in your .env file."})
            st.rerun()
            return
        
        # Create context from available data
        context = "You are an agricultural advisor helping farmers make decisions based on weather and soil data. "
        
        if st.session_state.weather_data:
            weather = st.session_state.weather_data
            context += f"Weather data for {weather['city']}: "
            context += f"Current temperature: {weather['current']['temp']}°F ({weather['current']['temp_c']:.2f}°C), "
            context += f"Humidity: {weather['current']['humidity']}%, "
            context += f"Precipitation: {weather['current']['precipitation']} inches, "
            context += f"Wind speed: {weather['current']['windspeed']} mph. "
            
            # Add forecast information
            if weather['forecast'] and len(weather['forecast']) > 0:
                context += "Weather forecast for the next 3 days: "
                for i, day in enumerate(weather['forecast'][:3]):
                    context += f"Day {i+1}: Temp {day['temp']}°F ({day['temp_c']:.2f}°C), "
                    context += f"Precipitation probability: {day['precipprob']}%, "
                    context += f"Conditions: {day['condition']}. "
        
        if st.session_state.soil_data:
            soil = st.session_state.soil_data
            context += "Soil data: "
            for key, value in soil.items():
                context += f"{key}: {value}, "
        
        if st.session_state.recommended_crop:
            context += f"Based on the soil and weather conditions, the recommended crop is: {st.session_state.recommended_crop}. "
        
        # Put together the complete prompt
        full_prompt = context + "\n\nUser question: " + user_input + "\n\nProvide helpful, detailed agricultural recommendations based on the available data. If specific data is missing, mention what would be useful to know to give better advice."
        
        # Call Gemini API
        with st.spinner("Thinking..."):
            try:
                response = query_gemini(gemini_apikey, full_prompt)
                response_text = response["candidates"][0]["content"]["parts"][0]["text"]
                
                # Add response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            except Exception as e:
                error_message = f"⚠️ Error getting response: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})
        
        # Rerun to update the chat display
        st.rerun()
    
    # Input for new messages with session state
    user_input = st.text_input(
        "Type your question here:", 
        key="user_input",
        value=st.session_state.input_value,
        on_change=lambda: setattr(st.session_state, 'input_value', st.session_state.user_input)
    )
    
    # Send button
    if st.button("Send", key="send_button"):
        clear_and_send()
    
    # Allow Enter key to send message
    if user_input and user_input != st.session_state.input_value:
        st.session_state.input_value = user_input
        clear_and_send()
    
    # Instructions
    with st.expander("Example Questions"):
        st.markdown("""
        - What crops would be suitable for my soil type?
        - Is it a good time to plant wheat based on the weather forecast?
        - How should I adjust irrigation given the predicted rainfall?
        - What fertilizers would you recommend based on my soil analysis?
        - Are there any weather concerns I should be aware of for the next week?
        - What soil amendments do I need based on my soil health card?
        - How should I prepare my field for growing the recommended crop?
        - What are the typical yields for the recommended crop?
        """)
    
    # Footer
    st.markdown("---")


with tab5:
    # API Configuration for Chatbot
    API_KEY = "AIzaSyCj5GEtpcIKrA23V37On_mKX2g8QuHoNtU"  
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

    # Load Pest Detection Model
    @st.cache_resource
    def load_pest_model():
        return tf.keras.models.load_model("pest_classifier.h5")

    pest_model = load_pest_model()

    pest_labels = ["Aphids", "ArmyWorm", "Beetle", "BollWorm", "GrassHopper", "Mites", "Mosquito", "SawFly", "Stem_Borer"]

    # Load Plant Disease Model
    @st.cache_resource
    def load_plant_model():
        return tf.keras.models.load_model("trained_model.h5", compile=False)

    plant_model = load_plant_model()

    plant_classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
                    'Corn___Common_rust_', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot',
                    'Grape___Esca(Black_Measles)', 'Grape___Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                    'Orange___Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                    'Pepper___Bacterial_spot', 'Pepper___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
                    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
                    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                    'Tomato___Spider_mites_Two-spotted_spider_mite', 'Tomato___Target_Spot',
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

    # Function to process pest image
    def predict_pest(img):
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)

        prediction = pest_model.predict(img_array)
        return pest_labels[np.argmax(prediction)]

    # Function to process plant disease image
    def predict_disease(img):
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = plant_model.predict(img_array)
        return plant_classes[np.argmax(prediction)]

    # Function to parse plant disease result
    def parse_plant_result(result):
        parts = result.split('___')
        plant_name = parts[0]
        disease_name = parts[1] if len(parts) > 1 else "Unknown"

        disease_name = re.sub(r'[()]', ' ', disease_name).replace('_', ' ')
        return plant_name, disease_name

    # Function to get recommendation from chatbot
    def get_recommendation(disease_or_pest, plant_name=None):
        # Build a minimized context from session state data
        context = ""
        
        if st.session_state.weather_data:
            weather = st.session_state.weather_data
            # Simplified weather data
            context += f"Weather: Temp {weather['current']['temp_c']:.1f}°C, Humidity {weather['current']['humidity']}%, "
        
        if st.session_state.soil_data:
            soil = st.session_state.soil_data
            # Extract only the most relevant soil properties for fertilizer decisions
            important_properties = ['pH', 'N', 'P', 'K', 'organic_matter']
            soil_context = ""
            for key in important_properties:
                if key in soil:
                    soil_context += f"{key}: {soil[key]}, "
            if soil_context:
                context += f"Soil: {soil_context}"
        
        if st.session_state.recommended_crop:
            context += f"Recommended crop: {st.session_state.recommended_crop}. "

        # Create the enhanced prompt with focus on fertilizers and pesticides
        prompt = f"You are an agricultural specialist. "
        
        if plant_name and disease_or_pest:
            prompt += f"I've identified {disease_or_pest} on {plant_name}. "
        else:
            prompt += f"I've identified {disease_or_pest}. "
            
        # Prioritize fertilizers and pesticides in the prompt
        prompt += "Provide a focused treatment plan with the following priority (keep your response brief and focused):\n\n"
        prompt += "1. FERTILIZER RECOMMENDATIONS (30% of response): Provide 2-3 specific fertilizer recommendations with application rates and timing to strengthen the plant against this issue.\n\n"
        prompt += "2. PESTICIDE/TREATMENT OPTIONS (30% of response): List 2-3 effective organic and chemical treatment options with application instructions.\n\n"
        prompt += "3. BRIEF OVERVIEW (20% of response): Very briefly explain the issue impact.\n\n"
        prompt += "4. PREVENTION TIPS (20% of response): 2-3 key prevention strategies.\n\n"
        
        # Add context if available
        if context:
            prompt += f"Local conditions: {context}"

        payload = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ],
            "generation_config": {
                "temperature": 0.6,
                "top_p": 1,
                "top_k": 40,
                "max_output_tokens": 500
            }
        }

        response = requests.post(
            f"{BASE_URL}?key={API_KEY}",
            headers={"Content-Type": "application/json"},
            json=payload
        )

        result = response.json()
        
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except KeyError:
            return "Error: Unable to generate recommendations. Please try again."

    # Streamlit UI
    st.title("🌿 Plant & Pest Solution Finder")

    col1, col2 = st.columns(2)

    with col1:
        st.header("📷 Upload Pest Image")
        pest_image = st.file_uploader("Choose an image for pest classification", type=["jpg", "png", "jpeg"])
        
        if pest_image:
            img = image.load_img(pest_image, target_size=(224, 224))
            st.image(img, caption="Uploaded Pest Image", use_container_width=True)
            pest_name = predict_pest(img)
            st.success(f"🦗 Pest Identified: {pest_name}")

    with col2:
        st.header("📷 Upload Plant Disease Image")
        disease_image = st.file_uploader("Choose an image for plant disease detection", type=["jpg", "png", "jpeg"])

        if disease_image:
            img = image.load_img(disease_image, target_size=(128, 128))
            st.image(img, caption="Uploaded Plant Image", use_container_width=True)
            plant_disease_result = predict_disease(img)
            plant_name, disease_name = parse_plant_result(plant_disease_result)
            st.success(f"🌱 Plant: {plant_name}")
            st.warning(f"🦠 Disease Identified: {disease_name}")

    # Determine what input to send to chatbot
    disease_or_pest = None
    identified_plant = None

    if 'pest_name' in locals() and 'disease_name' in locals() and 'plant_name' in locals():
        disease_or_pest = f"{disease_name} disease and {pest_name} infestation"
        identified_plant = plant_name
    elif 'pest_name' in locals():
        disease_or_pest = pest_name
        identified_plant = st.session_state.recommended_crop if st.session_state.recommended_crop else None
    elif 'disease_name' in locals() and 'plant_name' in locals():
        disease_or_pest = disease_name
        identified_plant = plant_name

    # Generate recommendations if we have a valid input
    if disease_or_pest:
        st.header("🧪 Fertilizer & Treatment Recommendations")
        with st.spinner("Generating solutions..."):
            recommendation = get_recommendation(disease_or_pest, identified_plant)
            
            # Display recommendations with improved formatting
            st.markdown(recommendation)
            
            # Add a download button for the recommendations
            recommendation_text = f"# Treatment Plan for {identified_plant if identified_plant else ''} affected by {disease_or_pest}\n\n"
            recommendation_text += recommendation
            
            st.download_button(
                label="💾 Download Treatment Plan",
                data=recommendation_text,
                file_name=f"treatment_plan_{disease_or_pest.replace(' ', '_')}.md",
                mime="text/markdown"
            )

with tab6:
    # API Configuration
    API_KEY = "AIzaSyCj5GEtpcIKrA23V37On_mKX2g8QuHoNtU"  # Replace with your actual Gemini API key
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

    WEATHER_API_KEY = "E5QXWDVJHBZAHWNGYLPD97S87"  # Replace with your actual Weather API key
    WEATHER_BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

    # Function to fetch weather data
    def fetch_weather_data(city):
        url = f"{WEATHER_BASE_URL}/{city}?unitGroup=us&key={WEATHER_API_KEY}&contentType=json"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error fetching weather data: HTTP {response.status_code}")
                return None
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return None

    # Function to extract relevant weather details
    def extract_weather_data(weather_data):
        current = weather_data.get("currentConditions", {})
        
        return {
            "city": weather_data.get("resolvedAddress", "Unknown location"),
            "description": weather_data.get("description", "No description available"),
            "date": weather_data["days"][0]["datetime"],
            "current": {
                "temp": current.get("temp"),
                "feelslike": current.get("feelslike"),
                "humidity": current.get("humidity"),
                "precipitation": current.get("precip", 0),
                "precipprob": current.get("precipprob", 0),
                "windspeed": current.get("windspeed"),
                "pressure": current.get("pressure"),
                "solarradiation": current.get("solarradiation"),
                "solarenergy": current.get("solarenergy"),
                "condition": current.get("conditions"),
                "description": current.get("description", weather_data.get("description", "No description available"))
            },
            "forecast": [
                {
                    "date": day.get("datetime"),
                    "temp": day.get("temp"),
                    "feelslike": day.get("feelslike"),
                    "humidity": day.get("humidity"),
                    "precipitation": day.get("precip", 0),
                    "precipprob": day.get("precipprob", 0),
                    "windspeed": day.get("windspeed"),
                    "pressure": day.get("pressure"),
                    "solarradiation": day.get("solarradiation"),
                    "solarenergy": day.get("solarenergy"),
                    "condition": day.get("conditions"),
                    "description": day.get("description", "No description available")
                }
                for day in weather_data.get("days", [])
            ]
        }

    # Function to display geofencing map
    def show_geofencing_map(latitude, longitude, radius=5):
        """Display an interactive map with a geofencing boundary."""
        
        # Create a folium map centered at the given location
        m = folium.Map(location=[latitude, longitude], zoom_start=12)

        # Add a marker for the location
        folium.Marker([latitude, longitude], popup="Farm Location", tooltip="Click for info").add_to(m)

        # Add a circular geofence (5 km by default)
        folium.Circle(
            location=[latitude, longitude],
            radius=radius * 1000,  # Convert km to meters
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.2
        ).add_to(m)

        # Display the map in Streamlit
        return m

    # Function to get farm activity recommendations using Gemini API
    def get_farm_recommendations(weather_data):
        prompt = f"""
        Based on the following weather data for {weather_data['city']}:
        
        - Temperature: {weather_data['current']['temp']}°F
        - Humidity: {weather_data['current']['humidity']}%
        - Precipitation: {weather_data['current']['precipitation']} inches
        - Precipitation Probability: {weather_data['current']['precipprob']}%
        - Wind Speed: {weather_data['current']['windspeed']} mph
        - Solar Radiation: {weather_data['current']['solarradiation']} W/m²
        - Solar Energy: {weather_data['current']['solarenergy']} MJ/m²
        - Condition: {weather_data['current']['condition']}
        
        Provide concise recommendations for:
        1. Sowing and harvesting schedules: Suitable crops and preventive measures for extreme weather, sowing based on the season (get season from current date) arrival of the monsoon (Kharif crops) or winter season (Rabi crops),rainfall patterns helps in deciding the right sowing time, especially for rain-fed crops.
        2. Irrigation: Best method based on the current weather, example :Drip irrigation or flood irrigation is adjusted based on humidity, temperature, and rainfall forecasts. if erratic rainfall prvent waterlogging and soil erosion by using drip irrigation, if high humidity and rainfall use flood irrigation.
        3. Pest & Fertilizer: Potential pest and disease outbreaks and best fertilizer application timing. Weather conditions directly impact pest and disease outbreaks (e.g., high humidity can cause fungal infections in crops), so use weather data to alert farmers about potential outbreaks and suggest preventive measures. Heavy rainfall can wash away fertilizers (causing wastage and leaching),Timing of fertilizer application is optimized based on rainfall levels so recommend the best time for fertilizer application based on the weather forecast.

        Keep the response **to the point** and at the end give **summary in bullet points** about what actions to be taken for farm-planning.
        dont assume any weather data, use the data provided in the prompt.
        """

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generation_config": {
                "temperature": 0.6,
                "top_p": 1,
                "top_k": 40,
                "max_output_tokens": 300
            }
        }

        response = requests.post(
            f"{BASE_URL}?key={API_KEY}",
            headers={"Content-Type": "application/json"},
            json=payload
        )

        result = response.json()

        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except KeyError:
            return "⚠️ Error: Unable to generate farm recommendations. Please try again."

    st.title("🌱 Farm Activity Planning using Localized Weather & Geofencing")

    # Initialize session state variables properly
    if 'weather_data' not in st.session_state:
        st.session_state['weather_data'] = None
    if 'specific_data' not in st.session_state:
        st.session_state['specific_data'] = None
    if 'farm_plan' not in st.session_state:
        st.session_state['farm_plan'] = None
    if 'latitude' not in st.session_state:
        st.session_state['latitude'] = 16.117223
    if 'longitude' not in st.session_state:
        st.session_state['longitude'] = 83.15954
    if 'data_loaded' not in st.session_state:
        st.session_state['data_loaded'] = False
    if 'radius' not in st.session_state:
        st.session_state['radius'] = 2

    # User input for location
    location = st.text_input("Enter Location (City, Village, or Coordinates):", "Delhi")

    # Create a two-column layout for main content and map controls
    main_col, control_col = st.columns([3, 1])

    with control_col:
        # Place the radius slider in the right column instead of sidebar
        st.subheader("Map Settings")
        radius = st.slider("Geofence Radius (km)", 
                          min_value=1, 
                          max_value=50, 
                          value=st.session_state['radius'], 
                          key="radius_slider")
        # Update session state with new radius value
        st.session_state['radius'] = radius

    with main_col:
        # Get weather data button
        if st.button("Get Farm Planner") or st.session_state['data_loaded']:
            # Only fetch data if we don't already have it or if the location changed
            if not st.session_state['data_loaded']:
                with st.spinner("Fetching weather data..."):
                    weather_data = fetch_weather_data(location)
                    
                    if weather_data:
                        st.session_state['weather_data'] = weather_data
                        st.session_state['specific_data'] = extract_weather_data(weather_data)
                        
                        # Set coordinates based on weather data
                        try:
                            st.session_state['latitude'] = weather_data.get("latitude", 16.117223)
                            st.session_state['longitude'] = weather_data.get("longitude", 83.15954)
                        except:
                            # Default coordinates if weather data doesn't contain them
                            st.session_state['latitude'] = 16.117223
                            st.session_state['longitude'] = 83.15954
                        
                        # Generate farm activity plan
                        with st.spinner("Generating farm plan using AI..."):
                            st.session_state['farm_plan'] = get_farm_recommendations(st.session_state['specific_data'])
                        
                        st.session_state['data_loaded'] = True
            
            # Display data if we have it
            if st.session_state['data_loaded'] and st.session_state['specific_data'] is not None:
                specific_data = st.session_state['specific_data']
                
                # Display weather details with improved UI
                st.subheader(f"Weather Data for {specific_data['city']}")
                st.write(f"**Date:** {specific_data['date']}")
                
                # Display weather metrics in columns
                col1, col2, col3 = st.columns(3)
                col1.metric("🌡 Temperature", f"{specific_data['current']['temp']}°F", delta=f"Feels like {specific_data['current']['feelslike']}°F")
                col2.metric("💧 Humidity", f"{specific_data['current']['humidity']}%")
                col3.metric("☔ Precipitation", f"{specific_data['current']['precipitation']} inches", delta=f"Probability {specific_data['current']['precipprob']}%")
                
                col4, col5, col6 = st.columns(3)
                col4.metric("🌬 Wind Speed", f"{specific_data['current']['windspeed']} mph")
                col5.metric("📏 Pressure", f"{specific_data['current']['pressure']} hPa")
                col6.metric("☀ Solar Energy", f"{specific_data['current']['solarenergy']} MJ/m²")
                
                st.info(f"**Weather Condition:** {specific_data['current']['condition']}")
                st.write(f"**Description:** {specific_data['current']['description']}")

                # Keep the forecast feature
                st.subheader("🔮 Forecast for the Next Few Days")
                for day in specific_data["forecast"][:3]:  # Show only next 3 days
                    st.write(f"📅 **{day['date']}**: {day['condition']} ({day['temp']}°F)")
            elif not st.session_state['data_loaded']:
                st.error("❌ Failed to fetch weather data. Please try again.")

    # Add Geofencing section (full width)
    if st.session_state['data_loaded'] and st.session_state['specific_data'] is not None:
        st.subheader("🌍 Geofencing & Real-Time Weather Tracking")
        
        # Create columns for the inputs - using session state to maintain values
        col1, col2 = st.columns(2)
        
        with col1:
            latitude = st.number_input("Enter Latitude", value=st.session_state['latitude'], format="%.6f", key="lat_input")
            st.session_state['latitude'] = latitude
        with col2:
            longitude = st.number_input("Enter Longitude", value=st.session_state['longitude'], format="%.6f", key="lng_input")
            st.session_state['longitude'] = longitude
        
        # Show the map with geofencing - regenerated each time radius changes
        m = show_geofencing_map(st.session_state['latitude'], st.session_state['longitude'], st.session_state['radius'])
        folium_static(m)
        
        # Display farm planning recommendations
        if st.session_state['farm_plan']:
            st.subheader("🌾 Farm Planning Recommendations")
            st.success("Farm plan generated successfully!")
            st.write(st.session_state['farm_plan'])
        
        # Add a refresh button to clear session state and fetch new data
        if st.button("Reset and Fetch New Data"):
            st.session_state['weather_data'] = None
            st.session_state['specific_data'] = None
            st.session_state['farm_plan'] = None
            st.session_state['data_loaded'] = False
            st.rerun()