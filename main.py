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
from models import LogisticRegression, NaiveBayes, KNearestNeighbors, ModelEvaluator, NeuralNetwork

def main():
    print("="*80)
    print("HR ANALYTICS: DỰ ĐOÁN KHẢ NĂNG THAY ĐỔI CÔNG VIỆC")
    print("="*80)
    
    # 1. LOAD DATA
    print("\n[1/6] Đang load dữ liệu...")
    processor = DataProcessor()
    
    try:
        # Load dữ liệu thô
        data, headers = processor.load_csv('data/raw/aug_train.csv')
        print(f"Đã load {data.shape[0]} samples với {data.shape[1]} features")
    except FileNotFoundError:
        print("Không tìm thấy file dữ liệu!")
        print("Vui lòng tải dataset từ Kaggle và đặt vào data/raw/aug_train.csv")
        return
    except Exception as e:
        print(f"Lỗi khi load file: {e}")
        return
    
    # 2. EXPLORATORY DATA ANALYSIS
    print("\n[2/6] Đang phân tích dữ liệu sơ bộ...")
    
    # Lấy target column
    try:
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
    except ValueError:
        print("Lỗi: Không tìm thấy cột 'target' trong dữ liệu!")
        return

    # 3. SPLIT & PREPROCESSING (NEW LOGIC)
    print("\n[3-4/6] Chia tập dữ liệu và Xử lý (Fit Train -> Transform Test)...")

    # A. Định nghĩa các nhóm features
    ordinal_features = ['relevent_experience', 'education_level', 'experience', 
                        'company_size', 'last_new_job']
    # 'company_type' nên là nominal vì không có thứ tự rõ ràng (Pvt Ltd vs NGO)
    nominal_features = ['gender', 'major_discipline', 'enrolled_university', 'company_type']
    numerical_features = ['city_development_index', 'training_hours']
    
    # B. Lấy danh sách tên cột features (bỏ target và enrollee_id)
    feature_names = [h for h in headers if h not in ['enrollee_id', 'target']]
    
    # C. Tách dữ liệu thô (Raw X) và Target (y)
    # get_columns_by_names giúp lấy đúng thứ tự cột theo feature_names
    X_raw = processor.get_columns_by_names(data, headers, feature_names)
    y = target
    
    # D. Chia Train/Test trên dữ liệu THÔ trước
    print(" -> Đang chia Train/Test set (80/20)...")
    X_train_raw, X_test_raw, y_train, y_test = processor.train_test_split(
        X_raw, y, test_size=0.2, random_state=42
    )
    print(f"    Raw Train shape: {X_train_raw.shape}")
    print(f"    Raw Test shape:  {X_test_raw.shape}")

    # E. Xử lý TRAIN SET (Vừa học params, vừa biến đổi)
    print("\n -> Đang xử lý TRAIN set (Fit & Transform)...")
    X_train, _ = processor.fit_transform_features(
        X_train_raw,         
        feature_names,       # Danh sách tên cột đóng vai trò là headers cho X_raw
        y_train,             
        ordinal_features,
        nominal_features,
        numerical_features
    )

    # F. Xử lý TEST SET (Chỉ biến đổi dựa trên params đã học)
    print("\n -> Đang xử lý TEST set (Transform only)...")
    X_test = processor.transform_features(
        X_test_raw,          
        feature_names,       
        ordinal_features,
        nominal_features,
        numerical_features
    )

    # G. Kiểm tra kết quả xử lý
    print(f"\nShape sau xử lý:")
    print(f" Train: {X_train.shape}")
    print(f" Test:  {X_test.shape}")
    
    # Hiển thị các Categories đã học được từ One-Hot Encoding
    print("\nCategories đã học (Nominal Features):")
    for feat, cats in processor.nominal_categories_map.items():
        print(f" - {feat}: {len(cats)} categories -> {cats}")
    
    # 5. TRAIN MODELS
    print("\n[5/6] Đang train models...")
    
    models = {
        'Logistic Regression': LogisticRegression(
            learning_rate=0.1, 
            n_iterations=3000, # Tăng iteration để đảm bảo hội tụ tốt hơn
            regularization='l2',
            lambda_reg=0.0001,
            random_state=42
        ),
        'Naive Bayes': NaiveBayes(),
        'KNN (k=5)': KNearestNeighbors(k=5, metric='euclidean'),
        'Neural Network (MLP)': NeuralNetwork(
            input_size=X_train.shape[1], # Tự động lấy số lượng features
            hidden_size=32,              # Số lượng nơ-ron lớp ẩn
            learning_rate=0.1,
            n_iterations=2000,
            random_state=42
        )
    }
    
    evaluator = ModelEvaluator()
    results = {}
    
    for model_name, model in models.items():
        print(f"\n  Training {model_name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)

        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
            
            # XỬ LÝ LINH HOẠT CHO CẢ 1D VÀ 2D
            if y_prob.ndim == 2:
                if y_prob.shape[1] == 2:
                    # Trường hợp có 2 cột [P(class=0), P(class=1)]
                    y_prob = y_prob[:, 1]
                elif y_prob.shape[1] == 1:
                    # Trường hợp có 1 cột (Neural Network) -> flatten
                    y_prob = y_prob.flatten()
            # Nếu đã là 1D thì giữ nguyên
        else:
            y_prob = None
        
        # Evaluate
        acc = evaluator.accuracy(y_test, y_pred)
        prec = evaluator.precision(y_test, y_pred)
        rec = evaluator.recall(y_test, y_pred)
        f1 = evaluator.f1_score(y_test, y_pred)
        roc_auc = 0.0
        if y_prob is not None:
            roc_auc = evaluator.roc_auc_score(y_test, y_prob)
        
        results[model_name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1-Score: {f1:.4f}")
    
    # 6. RESULTS
    print("\n[6/6] Kết quả cuối cùng:")
    print("\n" + "="*90)
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
    print("="*90)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<25} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f}"
              f"{metrics['roc_auc']:<12.4f}")
    
    print("="*90)
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
    print(f"\nBest Model: {best_model[0]} (F1-Score: {best_model[1]['f1_score']:.4f})")
    
    # 7. VISUALIZATION (optional)
    print("\n[Bonus] Đang tạo visualizations...")
    visualizer = DataVisualizer()
    
    try:
        # Create results directory
        os.makedirs('results/figures', exist_ok=True)
        
        # 1. Target Distribution
        fig1 = visualizer.plot_target_distribution(y)
        fig1.savefig('results/figures/target_distribution.png', dpi=300, bbox_inches='tight')
        print("Đã lưu target_distribution.png")
        
        # 2. Confusion Matrix (Best Model)
        best_model_obj = models[best_model[0]]
        y_pred_best = best_model_obj.predict(X_test)
        fig2 = visualizer.plot_confusion_matrix(y_test, y_pred_best)
        fig2.savefig('results/figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("Đã lưu confusion_matrix.png")
        
        # 3. Correlation Heatmap 
        # Lưu ý: Dùng X_train để vẽ heatmap, lấy 15 feature đầu tiên
        n_features_viz = min(15, X_train.shape[1])
        
        # Lấy tên features đã xử lý từ processor
        processed_names = processor.feature_names_processed
        
        corr_matrix = processor.compute_correlation_matrix(X_train[:, :n_features_viz])
        
        # Vẽ biểu đồ (cần đảm bảo list tên feature khớp độ dài)
        fig3 = visualizer.plot_correlation_heatmap(
            corr_matrix, 
            processed_names[:n_features_viz]
        )
        fig3.savefig('results/figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("Đã lưu correlation_matrix.png")
        
        print("\nTất cả visualizations đã được lưu vào results/figures/")
        
    except Exception as e:
        print(f"Không thể tạo visualizations: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("HOÀN THÀNH!")
    print("="*80)
    print("\nNext steps:")
    print("1. Kiểm tra thư mục results/figures/ để xem biểu đồ.")
    print("2. Điều chỉnh tham số models (learning rate, k, etc.) trong main.py để cải thiện kết quả.")
    # -------------------------------------------------------------------------
    # [PHẦN THAY THẾ] 4. GENERATE PROBABILITY SUBMISSION FILE - VECTORIZED
    # -------------------------------------------------------------------------
    print("\nĐang tạo file submission dạng xác suất (Probability)...")

    try:
        # 1. Load dữ liệu submission
        X_sub = np.load('data/processed/X_submission.npy')
        sub_ids = np.load('data/processed/submission_ids.npy', allow_pickle=True)
        
        print(f"Loaded submission data: {X_sub.shape}")
        
        # 2. Chọn model tốt nhất để dự đoán
        best_model_name = best_model[0]
        model_to_use = models[best_model_name] 
        print(f"Using model: {best_model_name} for prediction...")

        # 3. DỰ ĐOÁN XÁC SUẤT
        if hasattr(model_to_use, 'predict_proba'):
            y_sub_prob = model_to_use.predict_proba(X_sub)
            
            # Xử lý shape của output
            if y_sub_prob.ndim == 2 and y_sub_prob.shape[1] == 2:
                y_sub_prob = y_sub_prob[:, 1]
            elif y_sub_prob.ndim == 2 and y_sub_prob.shape[1] == 1:
                y_sub_prob = y_sub_prob.flatten()
                
        else:
            print("Model này không hỗ trợ xuất xác suất! Đang dùng nhãn 0/1 thay thế.")
            y_sub_prob = model_to_use.predict(X_sub).astype(float)

        # 4. LƯU FILE CSV - VECTORIZED (KHÔNG DÙNG LOOP)
        output_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'submission_proba.csv')
        
        # Vectorized: Convert sub_ids sang int
        sub_ids_int = sub_ids.astype(float).astype(int)
        
        # Vectorized: Format probabilities thành string với 4 chữ số thập phân
        y_sub_prob_str = np.char.mod('%.4f', y_sub_prob)
        
        # Vectorized: Kết hợp IDs và probabilities
        # Sử dụng np.column_stack để ghép 2 arrays
        output_data = np.column_stack([sub_ids_int.astype(str), y_sub_prob_str])
        
        # Ghi file sử dụng np.savetxt (HOÀN TOÀN VECTORIZED)
        np.savetxt(
            output_path,
            output_data,
            fmt='%s',  # String format
            delimiter=',',
            header='enrollee_id,target',
            comments=''  # Tắt comment prefix
        )
                
        print(f"Đã lưu file xác suất tại: {output_path}")
        
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file dữ liệu submission. Hãy chạy Preprocessing trước!")
    except Exception as e:
        print(f"Lỗi không mong muốn: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()