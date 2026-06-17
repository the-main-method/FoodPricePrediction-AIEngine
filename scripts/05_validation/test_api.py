import requests
import json

def test_prediction():
    url = "http://127.0.0.1:8000/predict"
    
    # Sample payload based on the PredictionRequest model in api/app.py
    payload = {
        "Food_Item": "maize",
        "Item_Type": "grain",
        "Category": "cereals",
        "Vendor_Type": "retail"
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            print("[SUCCESS]")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"[FAILED] Status code: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("[CONNECTION ERROR] Is the API running? Run 'uvicorn api.app:app --reload' in another terminal.")

if __name__ == "__main__":
    test_prediction()
