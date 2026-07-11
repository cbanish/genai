import requests
import os

from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """
    Get current weather for a city.
    """

    url = f"https://wttr.in/{city}?format=j1"

    response = requests.get(url, timeout=10)

    data = response.json()

    current = data["current_condition"][0]

    return f"""
City: {city}
Temperature: {current['temp_C']}°C
Feels Like: {current['FeelsLikeC']}°C
Humidity: {current['humidity']}%
Weather: {current['weatherDesc'][0]['value']}
Wind Speed: {current['windspeedKmph']} km/h
"""

agent = create_agent(
    model="mistral-small-latest",
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant."
)

city = input("Enter city name: ")


response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": f"What is the weather in {city}?"
            }
        ]
    }
)

print(response)




