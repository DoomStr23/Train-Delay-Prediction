import requests
import xml.etree.ElementTree as ET
import pandas as pd
import os

# API credentials
api_key = "96a9d41573dcc4ae5f89ee42770e0100"
headers = {
    "DB-Client-Id": "7182678e8cc6025ced2de2641313ecf7",
    "DB-Api-Key": api_key,
    "accept": "application/xml"
}

# Station: Köln Hbf (EVA number)
url = "https://apis.deutschebahn.com/db-api-marketplace/apis/timetables/v1/fchg/8000207"

# Make request
response = requests.get(url, headers=headers)

# Process response
if response.status_code == 200:
    try:
        root = ET.fromstring(response.text)
        data = []

        for s in root.findall('.//s'):
            station_eva = s.get('eva')

            # ARRIVALS
            for ar in s.findall('ar'):
                for m in ar.findall('m'):
                    data.append({
                        "station_eva": station_eva,
                        "direction": "arrival",
                        "train_id": m.get('id'),
                        "type": m.get('t'),
                        "timestamp": m.get('ts'),
                        "tts": m.get('ts-tts')
                    })

            # DEPARTURES
            for dp in s.findall('dp'):
                for m in dp.findall('m'):
                    data.append({
                        "station_eva": station_eva,
                        "direction": "departure",
                        "train_id": m.get('id'),
                        "type": m.get('t'),
                        "timestamp": m.get('ts'),
                        "tts": m.get('ts-tts')
                    })

        # Create DataFrame
        df = pd.DataFrame(data)
        print(df.head())

        # Save to CSV
        if not df.empty:
            file_exists = os.path.isfile("train_data_log.csv")
            df.to_csv("train_data_log.csv", mode='a', index=False, header=not file_exists)
            print("Data appended to train_data_log.csv")
        else:
            print("No data to save.")

    except ET.ParseError as e:
        print(f"XML Parse Error: {e}")
else:
    print(f"HTTP Error {response.status_code}: {response.text}")

