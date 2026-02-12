from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re

#defining the Flask app
app = Flask(__name__)
CORS(app)

model = joblib.load("model/spam_detector.pkl")
vectorizer = joblib.load("model/spam_vectorizer.pkl")

#Function to remove punctuation, numbers, and convert to lowercase
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    return text


#Defining the route for prediction
@app.route("/predict", methods=["POST"])


def spam_check():
    try:

        #Getting the input data
        data  = request.get_json()

        #Validating the input data    
        if not data or "message" not in data:
            return jsonify({"error": "Invalid input. 'message' field is required."}), 400

        text = data["message"]
        cleaned_text = clean_text(text)

        V_text = vectorizer.transform([cleaned_text])

        checker = model.predict(V_text)[0]

        spam_prob = model.predict_proba(V_text)[0][1]
        spam_prob = round(spam_prob * 100, 2)

        print("Spam" if checker == 1 else "Ham")

        return jsonify({
            "prediction": "Spam" if checker == 1 else "Ham",
            "confidence": spam_prob
        })

    #Handling exceptions
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
#Running the Flask app
if __name__ == "__main__":
    app.run(debug=True)