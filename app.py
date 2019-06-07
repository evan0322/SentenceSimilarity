#!flask/bin/python
from flask import Flask
app = Flask(__name__)
from flask import Flask
from flask import request
app = Flask(__name__)
from ml_manager import MLManager

app = Flask(__name__)
app.debug = True

mlmanager = MLManager('lite')

@app.route('/postjson', methods=['POST'])
def post():
    print(request.is_json)
    content = request.get_json()
    # print(content)
    sen1 = content['sen1']
    sen2 = content['sen2']

    col = mlmanager.get_STS_score_lite(sen1, sen2)
    return f'{col}'
app.run(host='0.0.0.0', port=5000)
