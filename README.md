# AI-Driven Agricultural Assistant

Welcome to the **AI-Driven Agricultural Assistant**, a Streamlit-based application designed to empower farmers with actionable insights for improving crop productivity and sustainability. Built for a hackathon, this app integrates AI-powered tools to forecast pest and disease risks, analyze soil conditions, optimize crop growth, and provide personalized agricultural advice.  

The project is deployed on **Hugging Face Spaces**, a cloud platform, making it accessible online without local setup.

## Project Overview

This application assists farmers by offering a comprehensive suite of tools to monitor weather, assess soil health, recommend crops, predict plant diseases and pests, and provide expert advice through an AI chatbot. The app is divided into multiple tabs, each serving a distinct purpose, with data flowing seamlessly across modules to enable informed decision-making. Additionally, users can download data in JSON format and graph images for better visualization and analysis.

---

## Features

### 1. Weather Forecast
- **Input**: Farmer enters their location (e.g., city or coordinates).
- **Output**:
  - Current weather conditions: temperature, humidity, wind speed, pressure, precipitation, and weather condition (e.g., sunny, rainy).
  - 14-day weather forecast for planning agricultural activities.
  - Location details.
- **Download**: Weather data available as JSON and graphical forecast images.
- **Storage**: Weather data stored in JSON format for use in subsequent modules.

### 2. Soil Health Card Reader
- **Input**: Farmer uploads an image of an official government-issued soil health card.
- **Output**:
  - Extracted data: pH, Nitrogen (N), Potassium (K), Phosphorus (P), organic carbon, micronutrients, soil type, farm size (acres), longitude, latitude, and irrigation type.
- **Download**: Soil data downloadable as JSON and nutrient-level graphs.
- **Storage**: Soil parameters saved in JSON format for integration with other features.

### 3. Crop Recommender
- **Input**: Uses JSON data from Weather Forecast and Soil Health Card Reader.
- **Output**: Suggests a list of suitable crops based on weather parameters (e.g., temperature, precipitation) and soil conditions (e.g., nutrient levels, pH).
- **Download**: Crop suggestions available as JSON and visual comparison charts.

### 4. Agricultural Advisor (Chatbot)
- **Functionality**: A Retrieval-Augmented Generation (RAG) model trained on data from weather, soil, and other agricultural sources.
- **Capabilities**:
  - Answers queries related to irrigation, fertilization (type and quantity), and soil management.
  - Provides tailored advice based on data collected from previous steps.
- **Use Case**: Acts as a virtual expert for farmers seeking real-time guidance.

### 5. Plant Disease and Pest Prediction
- **Input**: Farmer uploads images of crops showing signs of pests or diseases (supports single or dual inputs).
- **Process**:
  - Utilizes custom AI model, to identify pest types and plant diseases.
  - Combines image analysis with soil and weather data for context-aware insights.
- **Output**:
  - Identifies pest/disease type.
  - Suggests actionable treatments and future prevention strategies.
- **Download**: Prediction results and treatment suggestions available as JSON and annotated images.

---

## Smart Workflow (Flowchart Description)

The app follows an intelligent, interconnected workflow to deliver a seamless experience. Below is a textual representation of the smart flowchart:

1. **Start**: Farmer inputs location in the Weather Forecast tab.
   - **Output**: Weather data (current + 14-day forecast) stored in JSON, downloadable with graphs.
   - **Next**: Data flows to Crop Recommender and Agricultural Advisor.

2. **Soil Health Analysis & Geofencing**: Farmer uploads a soil health card image.
   - **Output**: Soil parameters extracted, stored in JSON, downloadable with nutrient graphs and geofenced map of that specific region.
   - **Next**: Data feeds into Crop Recommender and Agricultural Advisor.

3. **Crop Recommendation**: Combines weather and soil JSON data.
   - **Output**: List of recommended crops, downloadable as JSON and charts.
   - **Next**: Farmer can consult the Agricultural Advisor for further guidance.

4. **Agricultural Advisor**: RAG model processes all prior data (weather, soil, crops).
   - **Output**: Answers farmer queries and provides advice.
   - **Next**: Runs in parallel with other tabs as a support tool.

5. **Pest and Disease Prediction**: Farmer uploads crop images.
   - **Process**: MobileNet identifies pests/diseases, integrates with soil/weather data.
   - **Output**: Diagnosis and treatment suggestions, downloadable as JSON and images.
   - **Next**: Insights shared with Agricultural Advisor for follow-up queries.

**Data Flow**: JSON files act as a central repository, linking all tabs and enabling downloads for offline use.

---

## Technologies Used

- **Streamlit**: Frontend framework for building interactive tabs.
- **Python**: Core programming language for backend logic.
- **MobileNet**: Deep learning model for pest and disease image classification.
- **RAG (Retrieval-Augmented Generation)**: AI chatbot for agricultural advice.
- **JSON**: Data storage, transfer, and download format.
- **APIs**: Weather data retrieval (e.g., OpenWeatherMap or similar).
- **Image Processing**: Libraries like OpenCV/PIL for soil card and crop image analysis.
- **Hugging Face Spaces**: Cloud platform for deployment.

---

## Try It Out

The app is live on **Hugging Face Spaces**! Access it here:  
[**AI-Driven Agricultural Assistant on Hugging Face Spaces**](https://huggingface.co/spaces/<your-username>/<your-space-name>)  
*(Replace `<your-username>` and `<your-space-name>` with your actual Hugging Face details.)*

No local setup is required—just visit the link, explore the tabs, and download data (JSON files and graphs) as needed!

---

## Local Setup Instructions (Optional)

If you’d like to run the app locally:

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd ai-driven-agricultural-assistant
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **API Keys**:
   - Obtain an API key for weather data (e.g., OpenWeatherMap) and add it to a `.env` file:
     ```
     WEATHER_API_KEY=<your-api-key>
     ```

4. **Run the App**:
   ```bash
   streamlit run app.py
   ```

5. **Usage**:
   - Open your browser at `http://localhost:8501`.
   - Navigate through the tabs and download data/graph images.

---

## Hackathon Context

This project was developed for a hackathon to address real-world agricultural challenges using AI. Our goal is to provide farmers with an accessible, all-in-one tool—now deployed on Hugging Face Spaces—to improve decision-making, boost yields, and reduce risks from pests, diseases, and unpredictable weather.

---

## Future Enhancements

- Add real-time IoT sensor integration for soil and weather data.
- Support multilingual chatbot responses for wider accessibility.
- Include a cost-benefit analysis feature for crop recommendations.
- Enhance graph downloads with interactive visualizations.

##Access the bot here:
https://teamalpha-agriai.streamlit.app/

---

## Team

Built by Team Alpha for AgriAI Hackathon, February 2025.

