from flask import Flask, jsonify, request
from c125 import predictimage
app= Flask(__name__)

@app.route('/predict', methods=['POST'])
def home():
    image=request.files.get('digit')
    p=predictimage(image)
    return jsonify({'prediction': p}), 200

if __name__=='__main__':
    app.run(debug=True)