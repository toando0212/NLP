import numpy as np
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocess import process_20newsgroups

# --- Bước 1: Tải và xử lý dataset 20 Newsgroups ---
print("Đang tải và xử lý dataset 20 Newsgroups...")
X, y, vectorizer, target_names = process_20newsgroups()
print("Số lượng mẫu:", X.shape[0])
print("Số lượng lớp:", len(target_names))

# --- Bước 2: Chia dữ liệu thành tập train và test (giữ lại phân phối lớp) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Bước 3: Khởi tạo mô hình PassiveAggressiveClassifier ---
model_pa = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
print("Đang huấn luyện mô hình PassiveAggressiveClassifier...")
model_pa.fit(X_train, y_train)

# --- Bước 4: Đánh giá mô hình ---
train_pred = model_pa.predict(X_train)
test_pred = model_pa.predict(X_test)
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print("\nKết quả huấn luyện:")
print(f"Độ chính xác tập train: {train_acc:.2%}")
print(f"Độ chính xác tập test:  {test_acc:.2%}")
print("\nBáo cáo phân loại trên tập test:")
print(classification_report(y_test, test_pred, target_names=target_names))

# --- Bước 5: Lưu model và vectorizer ---
joblib.dump(model_pa, "topic_classifier_pa_20news.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer_20news.pkl")
print("Đã lưu model và vectorizer.")

# --- Bước 6: Tính Learning Curve với tqdm ---
train_sizes = np.linspace(0.1, 1.0, 10)
train_accuracies = []
test_accuracies = []

print("Calculating Learning Curve với PassiveAggressiveClassifier...")
for size in tqdm(train_sizes, desc="Learning Curve Progress"):
    if size == 1.0:
        X_train_part = X_train
        y_train_part = y_train
    else:
        X_train_part, _, y_train_part, _ = train_test_split(
            X_train, y_train, train_size=float(size), random_state=42, stratify=y_train
        )
    model_pa.fit(X_train_part, y_train_part)
    train_acc_part = accuracy_score(y_train_part, model_pa.predict(X_train_part))
    test_acc_part = accuracy_score(y_test, model_pa.predict(X_test))
    train_accuracies.append(train_acc_part)
    test_accuracies.append(test_acc_part)

plt.figure(figsize=(8,5))
plt.plot(train_sizes, train_accuracies, label="Train Accuracy", marker="o")
plt.plot(train_sizes, test_accuracies, label="Test Accuracy", marker="s")
plt.xlabel("Training data size ratio")
plt.ylabel("Accuracy")
plt.title("Learning Curve với PassiveAggressiveClassifier trên 20 Newsgroups")
plt.legend()
plt.grid(True)
plt.savefig("learning_curve_pa_20news.png")
plt.show()
