from flask import Flask, request
from flask_restful import Api, Resource
from datetime import datetime, date
import re
import tensorflow as tf
import keras
import numpy as np
import os
import json
import operator
from PIL import Image
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
from keras.preprocessing import image
app = Flask(__name__)
api = Api(app)

class Model:
    def __init__(self, modelFilePath):
        self.model_path = modelFilePath
        self.model = load_model(self.model_path)
        with open('cropClasses.json', 'r') as f:
            self.cropClasses = json.load(f)

    def preprocess(self, imageFilePath):
        image = tf.keras.preprocessing.image.load_img(imageFilePath, target_size=(224,224), interpolation="nearest")
        image_array = input_arr = keras.preprocessing.image.img_to_array(image)
        input_array = np.array([image_array])
        return input_array

    def determineTopXClasses(self, predictionArray, xClasses):
        predictDict = {}
        topClasses = []
        for idx, prediction in enumerate(predictionArray):
            predictDict[idx] = prediction
        for entries in list(sorted(predictDict.items(),
        key=operator.itemgetter(1),
        reverse=True))[0:xClasses]:
            topClasses.append(entries[0])
        return topClasses

    def determineClassNames(self, classArray):
        classNames = []
        for classNum in classArray:
            classNum = int(classNum)
            for key, value in self.cropClasses.items():
                if classNum == value:
                    classNames.append(key)

        if len(classNames) < len(classArray):
            return "one or more invalid class numbers"
        else:
            return classNames

    def determineClassName(self, classNum):
        classNum = int(classNum)
        for key, value in self.cropClasses.items():
            if classNum == value:
                return key
        return "key doesn't exist"

    def percentLikelihood(self, topClasses, predictionArray):
        percentLikelihood = []
        for _class in topClasses:
            percentLikelihood.append('{:.10f}'.format(predictionArray[_class]*100))
        print(percentLikelihood)
        return percentLikelihood

    def infer(self, imageFilePath = None):
        image = self.preprocess(imageFilePath)
        return self.model.predict(image)

# initialize some data prior to creating the API
cwd = os.getcwd()
MODEL_FILE_PATH = cwd + "\\EfficientNet.h5"
model = Model(MODEL_FILE_PATH)
with open('recourse.json', 'r') as f:
    recourseInfo = json.load(f)
model.determineClassName(0) #this forces the model to instantiate

class HelloWorld(Resource):
    def post(self, name):
        return {"data": "Hello World POST " + name}

    def get(self, name):
        return {"data": "Hello World GET " + name}

class Recourse(Resource):
    def get(self, defectName):
        recourse = recourseInfo[defectName]['recourse']
        prevention = recourseInfo[defectName]['prevention']
        return {"recourse": recourse, "prevention": prevention}

class ModelInfo(Resource):
    def get(self):
        return json.dumps(model.cropClasses)

class Image(Resource):
    def put(self):
        now = datetime.now()
        today = date.today()
        currentDate = today.strftime("%y-%m-%d")
        currentTime = now.strftime("%H-%M-%S")

        sentImageName = request.form['imageName']
        sentImage = request.files['image']

        pngPat = r"\.png"
        jpgPat = r"\.jpg"
        jpgPat2 = r"\.JPG"
        pngCheck = re.search(pngPat, sentImageName)
        jpgCheck = re.search(jpgPat, sentImageName)
        jpgCheck2 = re.search(jpgPat2, sentImageName)

        if pngCheck == None and jpgCheck == None and jpgCheck2 == None:
            return{"data": "INVALID FILE FORMAT. ONLY jpg OR png"}

        filetype = sentImageName.split('.')[-1]
        sentImageFileName = currentDate + '-' + currentTime + '.' + filetype
        sentImage.save(sentImageFileName)
        cwd = os.getcwd()
        filePath = cwd + "/" + sentImageFileName

        results = model.infer(filePath)[0]
        topClasses = model.determineTopXClasses(results, 3)
        topClassNames = model.determineClassNames(topClasses)

        os.remove(filePath)

        return{
        "data": "IMAGE " + sentImageName + " UPLOADED",
        "results" : json.dumps(results.tolist()),
        "classNo" : topClasses[0],
        "className": topClassNames[0],
        "topClasses": topClasses,
        "topClassNames": topClassNames,
        "topClassesPercent": model.percentLikelihood(topClasses, results)
          }

    def get(self, image):
        return{"data": "GET TEST SUCCESS " + image}

api.add_resource(HelloWorld, "/hello_world/<string:name>")
api.add_resource(Image, "/upload_image")
api.add_resource(Recourse, "/recourse/<string:defectName>")
api.add_resource(ModelInfo, "/model_info")



if __name__ == "__main__":
    app.run(debug=False)
