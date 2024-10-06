from flask import Flask, request, render_template
import cv2 as cv
from skimage.transform import resize
import pickle
import numpy as np
import base64

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

dog_cascade = cv.CascadeClassifier('cascade.xml')
cat_cascade = cv.CascadeClassifier('haarcascade_frontalcatface.xml')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    img = cv.imdecode(np.fromstring(image.read(), np.uint8), cv.IMREAD_COLOR)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cat_faces = cat_cascade.detectMultiScale(gray, 1.3, 5)

    resized_img = resize(img, (15, 15))
    resized_img = resized_img.flatten()

    resized_img = (resized_img * 255).astype('uint8')

    predict = model.predict([resized_img])
    prediction = "CAT" if predict == 1 else "DOG"

    _, buffer = cv.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)
    img_base64 = img_str.decode('utf-8')

    decoded_img = base64.b64decode(img_base64)
    np_arr = np.frombuffer(decoded_img, np.uint8)
    img_np = cv.imdecode(np_arr, cv.IMREAD_COLOR)

    for (x, y, w, h) in cat_faces:
        cv.putText(img_np, 'CAT', (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
        cv.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 4)

    _, buffer = cv.imencode('.jpg', img_np)
    img_base64_with_detection = base64.b64encode(buffer).decode('utf-8')

    return render_template('result.html', prediction=prediction, image=img_base64_with_detection)


if __name__ == '__main__':
    app.run(debug=True)
