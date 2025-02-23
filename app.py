from flask import Flask, request, render_template
import joblib
import numpy as np


from preprocess import clean_text  # Hàm clean_text có trong preprocess.py
from sklearn.datasets import fetch_20newsgroups

app = Flask(__name__)

# Load model và vectorizer đã lưu
model = joblib.load("topic_classifier_logreg_20news.pkl")
vectorizer = joblib.load("tfidf_vectorizer_20news.pkl")

# Tải target_names để ánh xạ từ số (nhãn gốc) thành tên ban đầu của 20 Newsgroups
newsgroups = fetch_20newsgroups(subset="all", remove=('headers', 'footers', 'quotes'))
target_names = newsgroups.target_names

# Tạo từ điển remapping: ánh xạ nhãn gốc thành các nhóm chủ đề lớn hơn, thân thiện hơn.
label_remap = {
    "alt.atheism": "Religion (Atheism)",
    "comp.graphics": "Computers (Graphics)",
    "comp.os.ms-windows.misc": "Computers (Windows)",
    "comp.sys.ibm.pc.hardware": "Computers (IBM PC)",
    "comp.sys.mac.hardware": "Computers (Mac)",
    "comp.windows.x": "Computers (Windows X)",
    "misc.forsale": "Classifieds",
    "rec.autos": "Automobiles",
    "rec.motorcycles": "Motorcycles",
    "rec.sport.baseball": "Baseball",
    "rec.sport.hockey": "Hockey",
    "sci.crypt": "Cryptography",
    "sci.electronics": "Electronics",
    "sci.med": "Medicine",
    "sci.space": "Space",
    "soc.religion.christian": "Christianity",
    "talk.politics.guns": "Politics (Guns)",
    "talk.politics.mideast": "Politics (Middle East)",
    "talk.politics.misc": "Politics",
    "talk.religion.misc": "Religion (Miscellaneous)"
}



# Vì một số nhãn có thể không nằm trong từ điển remap, ta sẽ giữ nguyên nếu không remap được.
def remap_label(orig_label):
    return label_remap.get(orig_label, orig_label)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction_main = None
    prediction_top = []
    text = ""
    if request.method == "POST":
        text = request.form["text"]
        # Tiền xử lý văn bản nhập vào
        cleaned_text = clean_text(text)
        X_input = vectorizer.transform([cleaned_text])

        # Lấy xác suất dự đoán cho tất cả các lớp
        probas = model.predict_proba(X_input)[0]
        # Lấy chỉ số của top 3 dự đoán (sắp xếp giảm dần)
        top_k = 3
        top_indices = np.argsort(probas)[::-1][:top_k]

        # Dự đoán chính là nhãn có xác suất cao nhất
        pred_idx = top_indices[0]
        pred_label = target_names[pred_idx]
        prediction_main = remap_label(pred_label)

        # Lấy thông tin cho top-k dự đoán: nhãn gốc, nhãn remap và xác suất
        for idx in top_indices:
            orig = target_names[idx]
            remapped = remap_label(orig)
            prob = probas[idx]
            prediction_top.append({"orig": orig, "remapped": remapped, "prob": prob})

    return render_template("index.html", prediction_main=prediction_main, prediction_top=prediction_top, text=text)


if __name__ == "__main__":
    app.run(debug=True)
