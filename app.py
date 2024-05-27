from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datetime as dt
import requests
import json

BASE_URL = "https://api.openweathermap.org/data/2.5/weather?"
API_KEY = open('api_key', 'r').read().strip()

with open('cities.txt', 'r', encoding='utf-8') as file:
    CITIES = [city.strip() for city in file.readlines()]

with open('chat_data.json', 'r', encoding='utf-8') as file:
    CHAT_DATA = json.load(file)['conversations']

def kelvin_to_celsius_fahrenheit(kelvin):
    celsius = kelvin - 273.15
    fahrenheit = celsius * (9/5) + 32
    return celsius, fahrenheit

def fetch_weather(city):
    url = BASE_URL + "appid=" + API_KEY + "&q=" + city.replace('_', ' ')
    response = requests.get(url).json()

    if response.get("cod") != 200:
        return {"city": city, "error": response.get("message", "Unknown error")}

    temp_kelvin = response['main']['temp']
    temp_celsius, temp_fahrenheit = kelvin_to_celsius_fahrenheit(temp_kelvin)
    feels_like_kelvin = response['main']['feels_like']
    feels_like_celsius, feels_like_fahrenheit = kelvin_to_celsius_fahrenheit(feels_like_kelvin)
    wind_speed = response['wind']['speed']
    description = response['weather'][0]['description']
    humidity = response['main']['humidity']
    sunrise_time = dt.datetime.utcfromtimestamp(response['sys']['sunrise'] + response['timezone'])
    sunset_time = dt.datetime.utcfromtimestamp(response['sys']['sunset'] + response['timezone'])

    weather_data = {
        "city": city,
        "temp_celsius": temp_celsius,
        "temp_fahrenheit": temp_fahrenheit,
        "feels_like_celsius": feels_like_celsius,
        "feels_like_fahrenheit": feels_like_fahrenheit,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "description": description,
        "sunrise_time": sunrise_time,
        "sunset_time": sunset_time
    }

    return weather_data

# Initialize chatbot
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", cls_token="<CLS>", mask_token="<MASK>", pad_token="<PAD>", sep_token="<SEP>")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    response = get_chat_response(msg)
    return jsonify(response)

def get_chat_response(text):
    text_lower = text.lower()
    matched_city = None
    
    for city in CITIES:
        if city.replace('_', ' ').lower() in text_lower:
            matched_city = city
            break

    if matched_city is None:
        # Check predefined conversations
        for conv in CHAT_DATA:
            if text_lower == conv["user"].lower():
                return conv["bot"]
        
        # Default response if no city is matched and no predefined conversation
        return "Sorry, I couldn't find the city you're looking for or understand your query."

    weather_data = fetch_weather(matched_city)

    if "error" in weather_data:
        return f"Error fetching weather for {matched_city}: {weather_data['error']}"

    if "temperature" in text_lower or "nhiệt độ" in text_lower:
        return f"Temperature in {matched_city.replace('_', ' ')}: {weather_data['temp_celsius']:.2f}°C or {weather_data['temp_fahrenheit']:.2f}°F"

    if "feels like" in text_lower or "cảm giác như" in text_lower:
        return f"Feels like in {matched_city.replace('_', ' ')}: {weather_data['feels_like_celsius']:.2f}°C or {weather_data['feels_like_fahrenheit']:.2f}°F"

    if "humidity" in text_lower or "độ ẩm" in text_lower:
        return f"Humidity in {matched_city.replace('_', ' ')}: {weather_data['humidity']}%"

    if "wind speed" in text_lower or "tốc độ gió" in text_lower:
        return f"Wind Speed in {matched_city.replace('_', ' ')}: {weather_data['wind_speed']} m/s"

    if "weather" in text_lower or "thời tiết" in text_lower:
        return (
            f"Temperature in {matched_city.replace('_', ' ')}: {weather_data['temp_celsius']:.2f}°C or {weather_data['temp_fahrenheit']:.2f}°F\n"
            f"Feels like in {matched_city.replace('_', ' ')}: {weather_data['feels_like_celsius']:.2f}°C or {weather_data['feels_like_fahrenheit']:.2f}°F\n"
            f"Humidity: {weather_data['humidity']}%\n"
            f"Wind Speed: {weather_data['wind_speed']} m/s\n"
            f"General Weather: {weather_data['description']}\n"
            f"Sunrise at {weather_data['sunrise_time']} local time\n"
            f"Sunset at {weather_data['sunset_time']} local time"
        )
    
    if "sunrise" in text_lower or "mặt trời mọc" in text_lower:
        return f"Sunrise in {matched_city.replace('_', ' ')} at {weather_data['sunrise_time']} local time."

    if "sunset" in text_lower or "mặt trời lặn" in text_lower:
        return f"Sunset in {matched_city.replace('_', ' ')} at {weather_data['sunset_time']} local time."

    # Chatbot response generation
    new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = new_user_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

if __name__ == '__main__':
    app.run(debug=True)
