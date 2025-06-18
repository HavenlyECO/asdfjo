import json
import requests

with open("production_game_state.json") as f:
    data = json.load(f)

resp = requests.post(
    "http://localhost:5000/api/advice",
    json=data,
)
print(resp.text)

