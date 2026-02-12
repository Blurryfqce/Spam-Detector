# This file is for testing the flask app
import joblib

message = input("Paste Message Here: ")

#Sending a POST request to the Flask app
import requests
response = requests.post("https://spam-detector-znos.onrender.com/predict", json={"message": message})
# print(response.json())

if response.json()["prediction"] == "Spam":
    print("The message is ", response.json()["confidence"], "% chance to be Spam.")
else:
    print("The message is ", 100 - response.json()["confidence"], "% chance to be real.")
    
