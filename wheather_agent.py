# import os
# import re
# import autogen
# import json
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import requests
# from dotenv import load_dotenv
# import google.generativeai as genai

# load_dotenv()

# app = FastAPI()

# class CityRequest(BaseModel):
#     city: str

# # Google Gemini Model Configuration
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # Ensure GOOGLE_API_KEY is set

# # Autogen LLM Config
# config_list = [
#     {
#         "model": "gemini-1.5-flash", #"gemini-pro",
#         "api_key": os.getenv("GOOGLE_API_KEY"),
#         "api_type": "google"
#     }
# ]

# llm_config = {
#     "config_list": config_list,
#     "seed": 42,
# }

# # Weather Agent (AutoGen function)
# def get_weather(city: str):
#     """Fetches weather data."""
#     api_key = os.getenv("WEATHER_API_KEY")
#     if not api_key:
#         return "Weather API key not found."
#     try:
#         response = requests.get(f"http://api.weatherstack.com/current?access_key={api_key}&query={city}")
#         response.raise_for_status()
#         data = response.json()
#         if 'current' in data and 'temperature' in data['current'] and 'weather_descriptions' in data['current']:
#             temperature = data['current']['temperature']
#             description = data['current']['weather_descriptions'][0]
#             wind_speed = data['current']["wind_speed"],
#             humidity = data['current']["humidity"],
#             precipitation= data['current']["precip"]
#             return f"Weather in {city}: {temperature}°C, {description}, wind_speed: {wind_speed}, humidity: {humidity}, precipitation: {precipitation}"
#         else:
#             return "Weather data incomplete or unavailable."
#     except requests.exceptions.RequestException as e:
#         return f"Weather Error: {e}"

# # Wikipedia Search Agent (AutoGen function)
# def search_wikipedia(city: str):
#     """Searches Wikipedia for city details."""
#     try:
#         response = requests.get(f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={city}&format=json")
#         response.raise_for_status()
#         data = response.json()
#         if data['query']['search']:
#             return data['query']['search'][0]['snippet']
#         else:
#             return "No Wikipedia results found."
#     except requests.exceptions.RequestException as e:
#         return f"Wiki Error: {e}"

# # Validation Agent (AutoGen Agent with Google Gemini)
# validation_agent = autogen.AssistantAgent(
#     name="validation_agent",
#     llm_config=llm_config,  # Directly pass the LLM configuration here
#     system_message="""
#     You are a validation expert. You will receive weather information and city details, 
#     and determine if they are accurate and comprehensive. 
#     A valid output should have a weather description, temperature, 
#     and at least 100 characters of city information. 
#     If the information is satisfactory, return a JSON object containing the weather and city info. 
#     If the information is not satisfactory, return 'None'.
#     """,
    
# )

# # Main Endpoint
# @app.post('/city_details')
# async def get_city_details(city_request: CityRequest):
#     """Orchestrates the agents and validation using AutoGen and Gemini."""
#     weather_result = get_weather(city_request.city)
#     wiki_result = search_wikipedia(city_request.city)

#     response = validation_agent.generate_reply(
#         messages=[{"role": "user", "content": f"Weather: {weather_result}\nCity Details: {wiki_result}"}],
#         sender=None,
#     )

#     content = response["content"]

#     # Extract the JSON string
#     json_string = re.search(r'```json\n(.*)\n```', content, re.DOTALL)
#     if json_string:
#         json_string = json_string.group(1)
#     else:
#         if content == "None":
#             raise HTTPException(status_code=500, detail="Failed to generate valid output.")
#         else:
#             raise HTTPException(status_code=500, detail="Invalid output format from validation agent.")

#     try:
#         return json.loads(json_string)
#     except json.JSONDecodeError:
#         raise HTTPException(status_code=500, detail="Invalid JSON from validation agent.")
    
# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host='127.0.0.1', port=8000)


import streamlit as st
import numpy as np
import pandas as pd
import os
import json
import re
import requests
from dotenv import load_dotenv
import google.generativeai as genai
import autogen

load_dotenv()

# Google Gemini Model Configuration
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Autogen LLM Config
config_list = [
    {
        "model": "gemini-1.5-flash",
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "api_type": "google"
    }
]

llm_config = {
    "config_list": config_list,
    "seed": 42,
}

# Weather Agent (AutoGen function)
def get_weather(city: str):
    """Fetches detailed weather data."""
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return "Weather API key not found."
    try:
        response = requests.get(f"http://api.weatherstack.com/current?access_key={api_key}&query={city}")
        response.raise_for_status()
        data = response.json()
        if 'current' in data:
            current = data['current']
            weather_details = {
                "temperature": current.get("temperature"),
                "description": current.get("weather_descriptions", [""])[0],
                "wind_speed": current.get("wind_speed"),
                "humidity": current.get("humidity"),
                "precipitation": current.get("precip"),
            }
            return json.dumps(weather_details)
        else:
            return "Weather data incomplete or unavailable."
    except requests.exceptions.RequestException as e:
        return f"Weather Error: {e}"

# Wikipedia Search Agent (AutoGen function)
def search_wikipedia(city: str):
    """Searches Wikipedia and cleans the result."""
    try:
        response = requests.get(f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={city}&format=json")
        response.raise_for_status()
        data = response.json()
        if data['query']['search']:
            snippet = data['query']['search'][0]['snippet']
            clean_snippet = re.sub(r'<[^>]+>', '', snippet)
            return clean_snippet
        else:
            return "No Wikipedia results found."
    except requests.exceptions.RequestException as e:
        return f"Wiki Error: {e}"

# Validation Agent (AutoGen Agent with Google Gemini)
validation_agent = autogen.AssistantAgent(
    name="validation_agent",
    llm_config=llm_config,
    system_message="""
    You are a validation expert. You will receive weather information and city details, 
    and determine if they are accurate and comprehensive. 
    A valid output should have a weather description, temperature, 
    and at least 100 characters of city information. 
    If the information is satisfactory, return a JSON object containing the weather and city info. 
    If the information is not satisfactory, return 'None'.
    """,
)

# Streamlit App
st.title("ClimaCity")

city_name = st.text_input("Enter City Name:")

if st.button("Get Details"):
    if city_name:
        weather_result = get_weather(city_name)
        wiki_result = search_wikipedia(city_name)
        print(f'weather result: {weather_result}')
        print(f'wiki result: {wiki_result}')

        response = validation_agent.generate_reply(
            messages=[{"role": "user", "content": f"Weather: {weather_result}\nCity Details: {wiki_result}"}],
            sender=None,
        )

        content = response["content"]

        # Extract the JSON string
        json_string = re.search(r'```json\n(.*)\n```', content, re.DOTALL)
        print(response)
        print(f'final json output: {json_string}')
        if json_string:
            json_string = json_string.group(1)
            # try:
            result = json.loads(json_string)
            print(f'final result: {result}')

            # Convert to Pandas DataFrame
            weather_df = pd.DataFrame([result["weather"]])
            city_df = pd.DataFrame([{"city": result["city_details"]}])

            st.subheader("Weather Details")
            st.dataframe(weather_df)

            st.subheader("City Details")
            st.dataframe(city_df)

            # except (json.JSONDecodeError, KeyError) as e:
            #     st.error(f"Error processing JSON: {e}")
        else:
            if content == "None":
                st.error("Failed to generate valid output.")
            else:
                st.error("Invalid output format from validation agent.")
    else:
        st.warning("Please enter a city name.")