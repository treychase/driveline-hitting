import certifi
import pandas as pd
from io import StringIO
import requests

def load_poi_metrics():
    url = "https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/main/baseball_hitting/data/poi/poi_metrics.csv"
    response = requests.get(url, verify=certifi.where())
    return pd.read_csv(StringIO(response.text))

def load_hittrax():
    url = "https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/main/baseball_hitting/data/poi/hittrax.csv"
    response = requests.get(url, verify=certifi.where())
    return pd.read_csv(StringIO(response.text))