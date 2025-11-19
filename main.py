"""
Main Script - Demo chạy nhanh toàn bộ pipeline
HR Analytics: Dự Đoán Khả Năng Thay Đổi Công Việc

Chạy script này để xử lý dữ liệu, train models và xem kết quả
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import DataProcessor
from visualization import DataVisualizer
from models import LogisticRegression, NaiveBayes, KNearestNeighbors, ModelEvaluator

def main():
    print("="*80)
    print("HR ANALYTICS: DỰ ĐOÁN KHẢ NĂNG THAY ĐỔI CÔNG VIỆC")
    print("="*80)
    
    # 1. LOAD DATA
    print("\n[1/6] Đang load dữ liệu...")
    processor = DataProcessor()
    
    try:
        data, headers = processor.load_csv('data/raw/aug_train.csv')
        print(f"Đã load {data.shape[0]} samples với {data.shape[1]} features")
    except FileNotFoundError:
        print("Không tìm thấy file dữ liệu!")
        print("Vui lòng tải dataset từ Kaggle và đặt vào data/raw/aug_train.csv")
        return
    
    # 2. EXPLORATORY DATA ANALYSIS
    print("\n[2/6] Đang phân tích dữ liệu...")
    
    # Lấy target column
    target_idx = headers.index('target')
    target = data[:, target_idx].astype(float)
    
    # Đếm phân phối target
    unique, counts = np.unique(target, return_counts=True)
    print(f"Phân phối target:")
    print(f"  - Class 0 (Không thay đổi): {counts[0]} ({counts[0]/len(target)*100:.1f}%)")
    print(f"  - Class 1 (Thay đổi): {counts[1]} ({counts[1]/len(target)*100:.1f}%)")
    
    # Check missing values
    missing_mask, missing_count = processor.check_missing_values(data)
    total_missing = np.sum(missing_count)
    print(f"Tổng missing values: {total_missing} ({total_missing/(data.shape[0]*data.shape[1])*100:.1f}%)")
    
    # 3. PREPROCESSING
    print("\n[3/6] Đang xử lý dữ liệu...")
    
    # Chọn features để train (bỏ enrollee_id và target)
    feature_names = [h for h in headers if h not in ['enrollee_id', 'target']]
    
    # Xác định categorical và numerical features
    numerical_features = ['city_development_index', 'training_hours']
    categorical_features = [f for f in feature_names if f not in numerical_features]
    
    X_processed = []
    feature_names_processed = []
    
    # Xử lý numerical features
    for feat in numerical_features:
        col_idx = headers.index(feat)
        col = processor.fill_missing_numerical(data[:, col_idx].copy(), strategy='median')
        col_normalized = processor.standardize_zscore(col)
        X_processed.append(col_normalized)
        feature_names_processed.append(feat)
    
    # Xử lý categorical features (chỉ lấy một số features quan trọng)
    important_categorical = ['gender', 'relevant_experience', 'enrolled_university', 
                            'education_level', 'major_discipline', 'experience']
    
    for feat in important_categorical:
        if feat in headers:
            col_idx = headers.index(feat)
            col = processor.fill_missing_categorical(data[:, col_idx].copy(), strategy='mode')
            col_encoded, mapping = processor.encode_categorical(col)
            col_normalized = processor.normalize_minmax(col_encoded.astype(float))
            X_processed.append(col_normalized)
            feature_names_processed.append(feat)
    
    X = np.column_stack(X_processed)
    y = target
    
    print(f"Dữ liệu sau xử lý: {X.shape[0]} samples, {X.shape[1]} features")
    
    # 4. SPLIT DATA
    print("\n[4/6] Đang chia train/test sets...")
    X_train, X_test, y_train, y_test = processor.train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 5. TRAIN MODELS
    print("\n[5/6] Đang train models...")
    
    models = {
        'Logistic Regression': LogisticRegression(
            learning_rate=0.01, 
            n_iterations=500,
            regularization='l2',
            lambda_reg=0.01,
            random_state=42
        ),
        'Naive Bayes': NaiveBayes(),
        'KNN (k=5)': KNearestNeighbors(k=5, metric='euclidean')
    }
    
    evaluator = ModelEvaluator()
    results = {}
    
    for model_name, model in models.items():
        print(f"\n  Training {model_name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        acc = evaluator.accuracy(y_test, y_pred)
        prec = evaluator.precision(y_test, y_pred)
        rec = evaluator.recall(y_test, y_pred)
        f1 = evaluator.f1_score(y_test, y_pred)
        
        results[model_name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        }
        
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1-Score: {f1:.4f}")
    
    # 6. RESULTS
    print("\n[6/6] Kết quả cuối cùng:")
    print("\n" + "="*80)
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("="*80)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<25} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f}")
    
    print("="*80)
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
    print(f"\nBest Model: {best_model[0]} (F1-Score: {best_model[1]['f1_score']:.4f})")
    
    # 7. VISUALIZATION (optional)
    print("\n[Bonus] Đang tạo visualizations...")
    visualizer = DataVisualizer()
    
    try:
        # Create results directory
        os.makedirs('results/figures', exist_ok=True)
        
        # Plot target distribution
        fig1 = visualizer.plot_target_distribution(y)
        fig1.savefig('results/figures/target_distribution.png', dpi=300, bbox_inches='tight')
        print("Đã lưu target_distribution.png")
        
        # Plot confusion matrix for best model
        best_model_obj = models[best_model[0]]
        y_pred_best = best_model_obj.predict(X_test)
        fig2 = visualizer.plot_confusion_matrix(y_test, y_pred_best)
        fig2.savefig('results/figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("Đã lưu confusion_matrix.png")
        
        # Plot correlation heatmap (subset of features)
        corr_matrix = processor.compute_correlation_matrix(X[:, :min(10, X.shape[1])])
        fig3 = visualizer.plot_correlation_heatmap(
            corr_matrix, 
            feature_names_processed[:min(10, len(feature_names_processed))]
        )
        fig3.savefig('results/figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("Đã lưu correlation_matrix.png")
        
        print("\nTất cả visualizations đã được lưu vào results/figures/")
        
    except Exception as e:
        print(f"⚠ Không thể tạo visualizations: {str(e)}")
    
    print("\n" + "="*80)
    print("HOÀN THÀNH!")
    print("="*80)
    print("\nNext steps:")
    print("1. Xem chi tiết trong notebooks/ để phân tích sâu hơn")
    print("2. Check results/figures/ để xem các biểu đồ")
    print("3. Thử điều chỉnh hyperparameters để cải thiện performance")
    print("\n")

if __name__ == "__main__":
    main()