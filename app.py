from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from main import meal_image_editor  # Assuming the script is saved as main.py

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Define the folder for input and output images
INPUT_FOLDER = 'images'
OUTPUT_FOLDER = 'edited'

# Ensure the folders exist
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Retrieve API key from environment
API_KEY = os.getenv('API_KEY')

# Test Endpoint
@app.route("/", methods=['GET'])
def test():
    test_response = """
        Hi :)
    """
    return test_response, 200

@app.route('/process_image', methods=['POST'])
def process_image():
    # Check for the API key in the request headers
    provided_api_key = request.headers.get('X-API-KEY')
    
    if provided_api_key != API_KEY:
        return jsonify({"error": "Unauthorized. Invalid API key."}), 401

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    # Save the image to the input folder
    input_path = os.path.join(INPUT_FOLDER, file.filename)
    file.save(input_path)

    # Extract the name without extension for saving the output
    name, _ = os.path.splitext(file.filename)

    try:
        # Process the image and get the GCS URL
        gcs_url = meal_image_editor(input_path, name)
        
        if gcs_url is None:
            return jsonify({"error": "Image processing failed"}), 500

        # Return the GCS URL of the processed image
        return jsonify({"message": "Image processed successfully.", "output_url": gcs_url}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is up and running!"}), 200

if __name__ == '__main__':
    app.run(debug=True)
