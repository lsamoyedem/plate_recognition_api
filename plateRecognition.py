from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
import numpy as np
import pytesseract
import imutils
import re
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import json

cred = credentials.Certificate("C:\\resource\\platerecognitionsd-firebase-adminsdk-hsdoj-13e6844f01.json")
default_app = firebase_admin.initialize_app(cred)

db = firestore.client()
collection = db.collection('logs')
teste =  { "plate":  "IML7489",
          "dateHour": "30 de novembro de 2022 00:34:14 UTC-3"
          }
         
#res = collection.add(teste)

resposta = collection.where('plate', '==', 'IML7393').get()
if resposta:
     print(resposta)




def detected(img):
     saida = ''
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     bfilter = cv2.bilateralFilter(gray, 11, 17, 17) # redução de ruido
     edged = cv2.Canny(bfilter, 30, 200) #detecta as bordas
     keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #detecta as contornos
     contours = imutils.grab_contours(keypoints) #fixa os contornos
     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
     location = None
     for contour in contours:
          approx = cv2.approxPolyDP(contour, 10, True)
          if len(approx) == 4:
               location = approx
               break
     if location is not None:          
          mask = np.zeros(gray.shape, np.uint8)

          cv2.drawContours(mask, [location], 0,255, -1) #desenha contornos
          cv2.bitwise_and(img, img, mask=mask)
          executar = True
          #corta a imagem
          try:
               (x,y) = np.where(mask==255)
               (x1, y1) = (np.min(x), np.min(y))
               (x2, y2) = (np.max(x), np.max(y))
               cropped_image = gray[x1:x2+1, y1:y2+1]
          except ValueError:
               executar = False 
               pass
      
          if executar:
               try:
                    image = cv2.resize(cropped_image, None, fx = 4, fy = 4,  interpolation = cv2.INTER_CUBIC)
                    image = cv2.GaussianBlur(image, (5, 5), 0)
                    image = cv2.threshold(image, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]   
                    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                    config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'
                    saida = pytesseract.image_to_string(image, lang='eng', config=config)
               except:
                    pass
     return saida


# Initialize the Flask application
app = Flask(__name__)


# route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    plate = detected(img)
    letras =  " ".join(re.findall("[a-zA-Z]+", plate))
    numeros = " ".join(re.findall("[0-9]+", plate))
    if (len(letras) == 3 or len(letras) == 4) and (len(numeros) == 3 or len(numeros) == 4):
        print(plate)
        response = {'plate': '{}'.format(plate)}
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype="application/json")
    else:    
         return Response(status=404, mimetype="application/json")

# start flask app
app.run(host="0.0.0.0", port=5000, threaded=True)