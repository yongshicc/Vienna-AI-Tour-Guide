import json

def load_personas(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

 def load_landmarks(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
