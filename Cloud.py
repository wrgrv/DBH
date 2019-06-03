from flask import Flask, render_template, Response
import cv2
from Frame import *

app = Flask(__name__)
#vc = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/frames')
def frames():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)