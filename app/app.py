import os
import io
import numpy as np

import keras
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.applications.xception import (
    Xception, preprocess_input, decode_predictions)
from keras.applications.vgg16 import (
   VGG16,
   preprocess_input,
   decode_predictions)
from keras import backend as K

from flask import Flask, request, redirect, url_for, jsonify, render_template, Response, session

app = Flask(__name__)
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = 'Uploads'

model = None
graph = None

def load_model():
    global model
    global graph

    model = keras.models.Sequential()
    for layer in keras.applications.vgg16.VGG16().layers[:-1]:
       model.add(layer)
       for layer in model.layers:
           layer.trainable = False
    model.add(keras.layers.core.Dense(4, activation='softmax'))
    model.load_weights('Re-trained.h5')

    graph = K.get_session().graph

load_model()

def prepare_image(img):
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    # return the processed image
    return img

@app.route('/')
def base():
    return render_template("index.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():

    data = {"success": False}
    if request.method == 'POST':
        
        uploaded_files = request.files.getlist("multi_upload")

        all_results = []

        # create directories to store each image type
        if os.path.exists('classification'):
            print('Already exists!')
        else:
            os.mkdir('classification')
            os.mkdir('classification/coyote')
            os.mkdir('classification/kit_fox')
            os.mkdir('classification/tortoise')
            os.mkdir('classification/gray_fox')
        
        for uploaded_file in uploaded_files:
            # create a path to the uploads folder
            filename = uploaded_file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(filepath)

            # Load the saved image using Keras and resize it to the Xception
            # format of 299x299 pixels
            image_size = (224, 224)
            im = keras.preprocessing.image.load_img(filepath,
                                                     target_size=image_size,
                                                     grayscale=False)

            # preprocess the image and prepare it for classification
            image = prepare_image(im)
            current_result = {}
            global graph
            with graph.as_default():
                preds = model.predict(image)
                
                
                #results = decode_predictions(preds)
                # print the results
                print(preds)

                #data["predictions"] = []

                # loop over the results and add them to the list of
                # returned predictions
                # for (imagenetID, label, prob) in preds[0]:
                #     r = {"label": label, "probability": float(prob)}
                #     data["predictions"].append(r)
                
                #v3y_true = test_labels.argmax(axis=1)
                pred = preds.argmax(axis=1)
                loc = pred[0]
                print(loc)
                probability = preds[0,loc]
                #data["predictions"] = v3y_pred
                #print(data["predictions"])
                #label = pred.argmax(axis=1)[0]
                if loc == 0:
                    label = 'Coyote'
                    uploaded_file.save(os.path.join('classification', 'coyote', filename))
                elif loc == 1:
                    label = 'Gray fox'
                    uploaded_file.save(os.path.join('classification', 'gray_fox', filename))
                elif loc == 2:
                    label = 'Kit fox'
                    uploaded_file.save(os.path.join('classification', 'kit_fox', filename))
                elif loc == 3:
                    label = 'Tortoise'
                    uploaded_file.save(os.path.join('classification', 'tortoise', filename))
                else:
                    label = "unknown"

                # indicate that the request was a success
                #data["success"] = True

            #label = data["predictions"][0]['label'].replace("_", " ").capitalize()
            #probability = round(data["predictions"][0]['probability'], 2)

            if probability >= 0.80:
                color = '#2ecc71'
            elif probability >= 0.60:
                color = '#f1c40f'
            else:
                color = '#e74c3c'
            current_result['Image'] = filename
            current_result['Prediction'] = label
            current_result['Probability'] = str(round(probability,3))
            current_result['Color'] = color
            all_results.append(current_result)


        session['all_results'] = all_results
        print(all_results)

        return render_template("index.html", data=all_results, scroll='results')


@app.route('/export')
def export():
    csv = 'Image,Prediction,Probability\n'
    all_results = session.get('all_results')
    for prediction in all_results:
        csv += f"{prediction['Image']},{prediction['Prediction']},{prediction['Probability']}\n"
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=predictions.csv"})

if __name__ == "__main__":
    app.run(debug=True)
