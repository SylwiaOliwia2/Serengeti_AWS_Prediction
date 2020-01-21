from flask import Flask, render_template
from predict_files import predict_files
import logging
import sys

app = Flask(__name__,  template_folder='templates')

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

## to call the app type  python app/app.py ANd open the lik displayed afterwards
@app.route('/')
def index():
    html = 'index.html'
    return render_template(html)

# predicting images
@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    app.logger.info('Running predict_files')
    predict_files()
    return {"result": "Done!"}


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port="5000")
