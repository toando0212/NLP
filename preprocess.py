import re
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
# Tải stopwords (chỉ cần chạy một lần)
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    """
    Tiền xử lý văn bản:
      - Chuyển về chữ thường.
      - Loại bỏ ký tự không phải chữ số và chữ cái.
      - Loại bỏ stopwords.
    """
    text = text.lower()
    # Loại bỏ ký tự không phải chữ cái, số hoặc khoảng trắng
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Tách từ và loại bỏ stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def process_20newsgroups(max_features=5000, stop_words='english'):
    """
    Tải dataset 20 Newsgroups, chuyển đổi văn bản thành ma trận TF-IDF.

    Trả về:
      - X: Ma trận TF-IDF.
      - y: Nhãn (dạng số từ 0 đến 19).
      - vectorizer: Đối tượng TfidfVectorizer đã được fit.
      - target_names: Danh sách tên chủ đề.
    """
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    X_text = newsgroups.data
    y = newsgroups.target
    target_names = newsgroups.target_names

    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    X = vectorizer.fit_transform(X_text)

    return X, y, vectorizer, target_names


if __name__ == '__main__':
    X, y, vectorizer, target_names = process_20newsgroups()
    print("Dataset processed:")
    print("Number of samples:", X.shape[0])
    print("Number of features:", X.shape[1])
    print("Target names:", target_names)
