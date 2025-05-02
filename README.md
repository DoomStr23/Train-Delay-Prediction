# Real-Time Public Transport Delay Prediction

This is a student project focused on building a system that predicts delays in public transport using real-time train timetable data from Deutsche Bahn. The main idea is to collect live arrival and departure data, clean it, engineer useful features, train a prediction model, and eventually visualize the results in a simple dashboard.

The project is built using Python and is split into small, manageable scripts that each handle one part of the pipeline: data collection, cleaning, feature engineering, model training, and visualization.

---

## 01_data_collection.py

This script fetches real-time train arrival and departure data from Cologne Hauptbahnhof (Köln Hbf) using Deutsche Bahn's timetable API. The data is saved to a file called `train_data_log.csv`.

It uses:
- `requests` for the API call
- `xml.etree.ElementTree` to parse the XML data
- `pandas` to store and save the data in CSV format

---

## Setup Instructions


