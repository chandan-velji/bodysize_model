import io
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
from keras.models import load_model

model = load_model('best_model.hdf5')

def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((64, 64))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img


def predict_result(img):
    return "Dog" if model.predict(img)[0][0] == 1 else "Cat"


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"
    
    file = request.files.get('file')

    if not file:
        return

    img_bytes = file.read()
    img = prepare_image(img_bytes)

    return jsonify(prediction=predict_result(img))
    

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')