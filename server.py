
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, jsonify, request, send_file
import base64
import zipfile
from werkzeug.utils import secure_filename
import keras
from backend_operations import image_predict, visualize_gradcam, visualize_lung_cancer
import shutil

# Define the input shape
input_shape = (122, 122, 3)

# Load the Keras model with the specified input shape
# model = keras.models.load_model("vgg_lc_v3.h5", compile= False,custom_objects={'input_shape': input_shape})
model = keras.models.load_model("vgg_lung_cancer.h5", compile= False,custom_objects={'input_shape': input_shape})

app = Flask(__name__)

@app.route('/')
def index():
    return "Inhouse Internship Project"

@app.route('/detect', methods=['POST'])
def cancer_detect():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'
    
    input_filename = secure_filename(file.filename)
    file_path = f"./uploads/{input_filename}"

    file.save(file_path)

    prob_scores, classes = image_predict(model, file_path)

    visualize_gradcam(model, file_path, target_class_idx=classes[0])

    heatmap = base64.b64encode(open('./uploads/heatmap.png','rb').read()).decode('ascii')
    output = base64.b64encode(open('./uploads/output.png','rb').read()).decode('ascii')


    return jsonify({'status':True, 'prob_scores': prob_scores, 'heatmap':heatmap, 'output': output, "classes":classes})


@app.route("/visualize", methods = ["POST"])
def visualizer():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    file.save('input_dir.zip')

    with zipfile.ZipFile('input_dir.zip', 'r') as zip_ref:
        zip_ref.extractall('./dcom-files')

    visualize_lung_cancer('./dcom-files',"lung_cancer_model.ply")

    shutil.rmtree('./dcom-files' )

    return send_file("lung_cancer_model.ply", as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)