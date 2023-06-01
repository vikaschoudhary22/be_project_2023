from flask import *
import numpy
import os

# from tensorflow.keras.preprocessing import image
# from keras.models import load_model

#doing ocr to scanning menu items
import easyocr
def recognize_text(img_path):
    '''loads an image and recognizes text.'''

    reader = easyocr.Reader(["en"])
    return reader.readtext(img_path)



#identifying food image using model
from tensorflow.keras.preprocessing import image
from keras.models import load_model
food_model = load_model('model_v1_inceptionV3.h5')
def classify(file_path):
    img_array = image.img_to_array(file_path)
    img_processed = numpy.expand_dims(img_array, axis=0)
    img_processed /= 255.

    prediction = food_model.predict(img_processed)

    index = numpy.argmax(prediction)
    # description = classes[index]
    return index

#translation
from translate_hi import decode_sequence_hindi
from translate_en import decode_sequence_english
from translate_ta import decode_sequence_tamil
from translate_mr import decode_sequence_marathi

app = Flask(__name__)

#home
@app.route('/')
def index():
    return "welcome to app"

#food identification
@app.route('/predict', methods=['GET', 'POST'])
def food_identify():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['img']
        f.save("img.jpg")
        # Make prediction
        result = classify("img.jpg")
        return result
    return None

#menu scanner
@app.route('/menu', menthods=['POST'])
def menu_scan_api():
    if request.method == 'POST':
        f = request.files['img']
        f.save('img_menu.jpg')
        result = recognize_text('img_menu.jpg')
        return result
    return None

#translation
@app.route("/translate", methods=["POST","GET"])
def translate():
    if request.method == "POST":
        # if "sentence" not in request.form:
        #     flash("NO sentence post")
        # elif request.form['sentence'] == '':
        #     flash("no sentence")
        # else:
        #     sent = request.form['sentence']
        #     translated = translate.translation(sent)
        #     return translated
        data = request.get_json()
        sent = data["sent"]
        lang = data["lang"]
        if lang == "hi":
            result = decode_sequence_hindi(sent)
            return result
        elif lang == "mr":
            result = decode_sequence_marathi(sent)
            return result
        elif lang == "ta":
            result = decode_sequence_tamil(sent)
            return result
        elif lang == "en":
            result = decode_sequence_english(sent)
            return result
    return None

# audio file
import pyttsx3
@app.route('/audio',methods=['POST'])
def audio_api():
    if request.method == 'POST':
        sentence = request.text
        result = audio_generation(sentence)
        # @after_this_request
        # os.remove('test.mp3')
        return send_file(result)

def audio_generation(sentence):
    engine = pyttsx3.init()
    engine.setProperty('rate', 100)
    engine.save_to_file(sentence, 'test.mp3')
    engine.runAndWait()
    return "test.mp3"


if __name__ == '__main__':
    app.run()