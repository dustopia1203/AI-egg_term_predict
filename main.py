import os
from egg_term_classify_service import classify_image
from flask import Flask, request, jsonify

app = Flask(__name__)

# Dev test only
@app.route('/', methods=['GET'])
def index():
    return "Hello world!"

@app.route('/api/egg-term/predict', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if image and image.filename.lower().endswith('.jpg'):
        image_path = os.path.join('uploads', image.filename)
        image.save(image_path)
        label = classify_image(image_path)
        return jsonify({"label": label})

    return jsonify({"error": "Invalid file format, only .jpg is allowed"}), 400

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)