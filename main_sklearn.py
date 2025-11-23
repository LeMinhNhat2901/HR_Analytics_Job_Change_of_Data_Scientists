"""
Main Script - Benchmark với Scikit-Learn
HR Analytics: Dự Đoán Khả Năng Thay Đổi Công Việc

Script này sử dụng thư viện chuẩn Scikit-Learn để so sánh kết quả
với mô hình tự viết (NumPy implementation).
"""

import pandas as pd
import numpy as np
import os

# Các module xử lý dữ liệu của Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Các mô hình
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def main():
    print("="*80)
    print("BENCHMARK: SCIKIT-LEARN IMPLEMENTATION")
    print("="*80)

    # 1. LOAD DATA (Dùng Pandas cho tiện lợi)
    print("\n[1/5] Loading Data (Pandas)...")
    try:
        df_train_full = pd.read_csv('data/raw/aug_train.csv')
        df_submission = pd.read_csv('data/raw/aug_test.csv')
        print(f"Train shape: {df_train_full.shape}")
        print(f"Test shape:  {df_submission.shape}")
    except FileNotFoundError:
        print("Error: Không tìm thấy file dữ liệu csv!")
        return

    # 2. PREPROCESSING PIPELINE SETUP
    # Sklearn cho phép tạo Pipeline xử lý tự động cực mạnh
    print("\n[2/5] Setting up Sklearn Pipelines...")

    # Phân loại cột
    target_col = 'target'
    
    # Ordinal: Có thứ tự (Scikit-Learn cần biết thứ tự này nếu muốn chính xác 100%, 
    # nhưng để nhanh ta để nó tự học hoặc dùng OrdinalEncoder đơn giản)
    ordinal_cols = ['relevent_experience', 'education_level', 'experience', 
                    'company_size', 'last_new_job']
    
    # Nominal: Không thứ tự -> OneHot
    nominal_cols = ['gender', 'major_discipline', 'enrolled_university', 'company_type']
    
    # Numerical
    numerical_cols = ['city_development_index', 'training_hours']

    # --- Định nghĩa các bước xử lý cho từng nhóm cột ---
    
    # 1. Xử lý số: Điền median -> Chuẩn hóa
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Xử lý Ordinal: Điền 'missing' -> Mã hóa số (0, 1, 2)
    # handle_unknown='use_encoded_value' giúp tránh lỗi khi gặp giá trị lạ
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # 3. Xử lý Nominal: Điền 'missing' -> One-Hot Encoding
    # handle_unknown='ignore' giúp bỏ qua các category lạ trong tập test
    nominal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
    ])

    # Gộp tất cả lại thành 1 bộ xử lý tổng (ColumnTransformer)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('ord', ordinal_transformer, ordinal_cols),
            ('nom', nominal_transformer, nominal_cols)
        ])

    # 3. SPLIT DATA
    print("\n[3/5] Splitting Data...")
    X = df_train_full.drop(columns=['enrollee_id', target_col])
    y = df_train_full[target_col]

    # Chia 80-20
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Train set: {X_train.shape}")
    print(f"Val set:   {X_val.shape}")

    # 4. TRAIN & EVALUATE
    print("\n[4/5] Training & Benchmarking Models...")
    
    # Định nghĩa các model cần test
    # Lưu ý: Sklearn LogisticRegression mặc định có L2 regularization
    models = {
        'Sklearn Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Sklearn Naive Bayes': GaussianNB(),
        'Sklearn KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'Sklearn MLP (Neural Net)': MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
    }

    results = {}

    print("\n" + "="*110)
    print(f"{'Model':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
    print("="*110)

    for name, model in models.items():
        # Tạo Pipeline: Xử lý dữ liệu -> Model
        # Đây là điểm mạnh của Sklearn: code cực gọn
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        
        # Train
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1] # Lấy xác suất class 1
        
        # Metrics
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_prob)
        
        print(f"{name:<30} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {auc:<10.4f}")
        
        results[name] = {'model': clf, 'f1': f1}

    print("="*110)

    # 5. PREDICT SUBMISSION (Best Model)
    print("\n[5/5] Generating Submission for Best Sklearn Model...")
    
    # Tìm model có F1 cao nhất
    best_name = max(results, key=lambda k: results[k]['f1'])
    best_model = results[best_name]['model']
    print(f"Best Model: {best_name}")

    # Dự đoán trên tập test thật (aug_test.csv)
    # Pipeline sẽ tự động xử lý Missing values, Encoding, Scaling cho tập test!
    X_sub = df_submission.drop(columns=['enrollee_id'])
    sub_ids = df_submission['enrollee_id']
    
    y_sub_pred = best_model.predict(X_sub)
    
    # Save
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission_sklearn.csv'
    
    submission_df = pd.DataFrame({
        'enrollee_id': sub_ids,
        'target': y_sub_pred.astype(int)
    })
    submission_df.to_csv(submission_path, index=False)
    
    print(f"Saved Sklearn submission to: {submission_path}")
    print("\nDone.")

if __name__ == "__main__":
    main()