# Code by AkinoAlice@Tyrant_Rex

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from herb_classifier import ChineseHerbClassificationModel
from PIL import Image

import os

app = Flask(__name__)

ALLOWED_EXTENSIONS = {"jpg", "jpg", "jpeg", "gif", "png"}
CLASSIFIER = ChineseHerbClassificationModel()
CLASSIFIER.load_model(path="./model/chinese_herb_classifier.pth")


def predict_img(target_path: str = "./test/test.jpg") -> tuple[str, float]:
    test_image = Image.open(target_path).convert("RGB")

    result, idx = CLASSIFIER.classifier(test_image)
    assert result, "None result in classifier"

    return result, idx


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

        result, idx = predict_img(
            target_path="./static/uploads/webcam_image.png")

        table = {
            "Angelica sinensis": "當歸",
            "Chuanxiong": "川芎",
            "Eucommia ulmoides": "杜仲",
            "Ginseng": "黨蔘",
            "Wolfberry": "枸杞",
        }

        return jsonify({
            "success": "file successfully uploaded",
            "filename": filename,
            "predictions": table[result]
        }), 200
    else:
        return jsonify({"error": "file_ type not allowed"}), 400


if __name__ == "__main__":
    app.run(debug=True)
