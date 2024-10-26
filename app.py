# Code by AkinoAlice@Tyrant_Rex

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from herb_classifier import Classifier

import os
app = Flask(__name__)

ALLOWED_EXTENSIONS = {"jpg", "jpg", "jpeg", "gif", "png"}
CLASSIFIER = Classifier()


def allowed_file_(file_name):
    return "." in file_name and file_name.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file_():
    if "file" not in request.files:
        return jsonify({"error": "No file_ part in the request"}), 400
    file_ = request.files["file"]
    if file_.filename == "":
        return jsonify({"error": "No selected file_"}), 400

    if file_ and allowed_file_(file_.filename):
        filename = secure_filename(file_.filename)

        if not os.path.exists("./static/uploads"):
            os.mkdir("./static/uploads")

        file_.save(os.path.join("./static/uploads", filename))

        _, result = CLASSIFIER.predict_img(
            model_path="./model/model.h5", target_path="./static/uploads/webcam_image.png")

        print(result)

        return jsonify({
            "success": "file successfully uploaded",
            "filename": filename,
            "predictions": result
        }), 200
    else:
        return jsonify({"error": "file_ type not allowed"}), 400


if __name__ == "__main__":
    app.run(debug=True)
