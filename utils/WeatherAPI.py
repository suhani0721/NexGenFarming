import requests
from langsmith import traceable
class WeatherAPI:

    @staticmethod
    @traceable(name="Weather API - Get Coordinates")
    def get_coordinates(location):
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {"name": location, "count": 1}

        response = requests.get(geo_url, params=geo_params)
        data = response.json()

        if "results" not in data:
            return None, None

        lat = data["results"][0]["latitude"]
        lon = data["results"][0]["longitude"]

        return lat, lon

    @staticmethod
    @traceable(name="Weather API - Get Current Weather")
    def get_weather(location):
        latitude, longitude = WeatherAPI.get_coordinates(location)

        if latitude is None:
            return "Location not found."

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current_weather": True
        }

        response = requests.get(url, params=params)
        data = response.json()

        return data.get("current_weather", "Weather data unavailable")
