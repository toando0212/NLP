import numpy as np
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from preprocess import process_20newsgroups

# --- Bước 1: Tải và xử lý dataset 20 Newsgroups ---
print("Loading and processing 20 Newsgroups dataset...")
X, y, vectorizer, target_names = process_20newsgroups()
print("Number of samples:", X.shape[0])
print("Number of classes:", len(target_names))

# --- Bước 2: Chia dữ liệu thành tập train và test (stratify để giữ phân phối lớp) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Bước 3: Khởi tạo mô hình LogisticRegression ---
model = LogisticRegression(
    solver='saga',
    n_jobs=-1,
    random_state=42,
    max_iter=2000,
    verbose=0  # Bật thông báo quá trình huấn luyện nếu cần
)

# --- Bước 4: Huấn luyện mô hình ---
print("Training LogisticRegression model...")
model.fit(X_train, y_train)

# --- Bước 5: Đánh giá mô hình ---
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print("\nTraining Results:")
print(f"Train Accuracy: {train_acc:.2%}")
print(f"Test Accuracy:  {test_acc:.2%}")
print("\nClassification Report on Test Set:")
print(classification_report(y_test, test_pred, target_names=target_names))

# --- Bước 6: Lưu model và vectorizer ---
joblib.dump(model, "topic_classifier_logreg_20news.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer_20news.pkl")
print("Model and vectorizer saved.")

# --- Bước 7: Tính Learning Curve với tqdm ---
train_sizes = np.linspace(0.1, 1.0, 10)
train_accuracies = []
test_accuracies = []

print("Computing Learning Curve...")
for size in tqdm(train_sizes, desc="Learning Curve Progress"):
    if size == 1.0:
        X_train_part = X_train
        y_train_part = y_train
    else:
        X_train_part, _, y_train_part, _ = train_test_split(
            X_train, y_train, train_size=float(size), random_state=42, stratify=y_train
        )
    model.fit(X_train_part, y_train_part)
    train_acc_part = accuracy_score(y_train_part, model.predict(X_train_part))
    test_acc_part = accuracy_score(y_test, model.predict(X_test))
    train_accuracies.append(train_acc_part)
    test_accuracies.append(test_acc_part)

plt.figure(figsize=(8,5))
plt.plot(train_sizes, train_accuracies, label="Train Accuracy", marker="o")
plt.plot(train_sizes, test_accuracies, label="Test Accuracy", marker="s")
plt.xlabel("Training Data Size Ratio")
plt.ylabel("Accuracy")
plt.title("Learning Curve with LogisticRegression on 20 Newsgroups")
plt.legend()
plt.grid(True)
plt.savefig("learning_curve_logreg_20news.png")
plt.show()
