
# Weather Data Analysis using OpenWeatherMap API

# 1. Import libraries
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. API configuration
api_key = "2df78213121eee78908471aea7b5c77e"
city = "Pune,IN"

url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"

# 3. Get weather data
response = requests.get(url)
data = response.json()

if response.status_code != 200:
    print("Error:", data.get("message", "Unable to fetch weather data"))
else:

    # 4. Extract weather information
    weather_records = []

    for item in data['list']:
        weather_records.append({
            'Datetime': item['dt_txt'],
            'Temperature': item['main']['temp'],
            'Humidity': item['main']['humidity'],
            'WindSpeed': item['wind']['speed'],
            'Precipitation': item.get('rain', {}).get('3h', 0)
        })

    # 5. Create DataFrame
    df = pd.DataFrame(weather_records)

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.fillna(0, inplace=True)

    print("Weather Data:")
    print(df.head())

    # 6. Daily average
    daily_avg = df.resample('D', on='Datetime').mean()
    print("\nDaily Average:")
    print(daily_avg)

    # 7. Temperature graph
    plt.figure(figsize=(10, 5))
    plt.plot(df['Datetime'], df['Temperature'])
    plt.title("Temperature Forecast")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 8. Temperature vs Humidity
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Temperature', y='Humidity', data=df)
    plt.title("Temperature vs Humidity")
    plt.show()

    # 9. Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.select_dtypes(include='number').corr(),
                annot=True, cmap='YlGnBu')
    plt.title("Weather Correlation")
    plt.show()
