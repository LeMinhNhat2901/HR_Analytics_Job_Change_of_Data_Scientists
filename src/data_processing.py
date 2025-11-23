"""
Data Processing Module - CHỈ SỬ DỤNG NUMPY
Module xử lý dữ liệu cho HR Analytics Job Change Prediction
"""
"""
Data Processing Module - CHỈ SỬ DỤNG NUMPY
Module xử lý dữ liệu chuẩn cho HR Analytics Job Change Prediction
"""

import numpy as np

class DataProcessor:
    """Class xử lý dữ liệu sử dụng NumPy"""
    
    def __init__(self):
        self.feature_names = []
        
        # --- CÁC KHO CHỨA THAM SỐ (STATE) ---
        # Lưu giá trị fill missing (Mean/Median/Mode) của Train
        self.imputation_values = {} 
        # Lưu mapping cho Ordinal features
        self.ordinal_mappings = {}
        # Lưu danh sách categories cho Nominal features
        self.nominal_categories_map = {}
        # Lưu min/max hoặc mean/std cho Numerical features
        self.numerical_params = {}
        # Lưu tên các feature sau khi xử lý xong
        self.feature_names_processed = []
        
    # =========================================================================
    # 1. BASIC I/O & UTILS
    # =========================================================================
    
    def load_csv(self, filepath, delimiter=','):
        """Đọc file CSV và trả về numpy array + headers"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                headers = f.readline().strip().split(delimiter)
            
            data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                next(f)  # Skip header
                for line in f:
                    row = line.strip().split(delimiter)
                    data.append(row)
            
            self.feature_names = headers
            return np.array(data, dtype=object), headers
        except Exception as e:
            print(f"Lỗi đọc file: {e}")
            return None, None
    
    def get_columns_by_names(self, data, headers, col_names):
        """Lấy nhiều cột theo danh sách tên"""
        indices = [headers.index(name) for name in col_names if name in headers]
        return data[:, indices]
    
    def train_test_split(self, X, y, test_size=0.2, random_state=None):
        """Chia dữ liệu Train/Test"""
        if random_state is not None:
            np.random.seed(random_state)
        
        n_samples = X.shape[0]
        n_test = int(n_samples * test_size)
        
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

    # =========================================================================
    # 2. LOW-LEVEL PROCESSING FUNCTIONS (STATEFUL)
    # =========================================================================
    def check_missing_values(self, data):
        """
        Kiểm tra missing values trong dữ liệu
        
        Returns:
            missing_mask: boolean array đánh dấu missing values
            missing_count: số lượng missing values mỗi cột
        """
        # Xác định missing values (empty string, 'nan', 'NaN', None)
        missing_mask = np.zeros(data.shape, dtype=bool)
        
        for i in range(data.shape[1]):
            col = data[:, i]
            mask = np.array([
                (val == '' or val == 'nan' or val == 'NaN' or 
                 val is None or str(val).lower() == 'nan')
                for val in col
            ])
            missing_mask[:, i] = mask
        
        missing_count = np.sum(missing_mask, axis=0)
        return missing_mask, missing_count

    def fill_missing_categorical(self, column, fill_value=None, strategy='mode'):
        """
        Điền missing values (Stateful).
        - Nếu fill_value là None (Train phase): Tự tính mode và trả về.
        - Nếu fill_value có giá trị (Test phase): Dùng giá trị đó để điền.
        """
        # Tạo mask cho các giá trị thiếu
        mask = np.array([
            val in ['', 'nan', 'NaN', None] or str(val).lower() == 'nan' 
            for val in column
        ])
        
        # Nếu chưa có giá trị fill (Train phase), hãy tính toán nó
        if fill_value is None:
            if strategy == 'mode':
                valid_values = column[~mask]
                if len(valid_values) > 0:
                    unique, counts = np.unique(valid_values, return_counts=True)
                    fill_value = unique[np.argmax(counts)]
                else:
                    fill_value = 'Unknown'
            else:
                fill_value = 'Unknown'
        
        # Điền giá trị
        column_filled = column.copy()
        column_filled[mask] = fill_value
        
        return column_filled, fill_value

    def fill_missing_numerical(self, column, fill_value=None, strategy='median'):
        """
        Điền missing numerical (Stateful).
        Trả về: (cột đã điền, giá trị fill đã dùng)
        """
        # Tạo mask
        mask = np.array([
            val in ['', 'nan', 'NaN', None] or str(val).lower() == 'nan' 
            for val in column
        ])
        
        # Chuyển sang float an toàn
        col_float = np.zeros(len(column), dtype=float)
        for i, val in enumerate(column):
            if not mask[i]:
                try:
                    col_float[i] = float(val)
                except:
                    mask[i] = True # Nếu lỗi parse thì coi là missing
        
        # Tính toán giá trị fill nếu chưa có (Train phase)
        if fill_value is None:
            valid_values = col_float[~mask]
            if len(valid_values) > 0:
                if strategy == 'mean':
                    fill_value = np.mean(valid_values)
                elif strategy == 'median':
                    fill_value = np.median(valid_values)
                else:
                    fill_value = 0.0
            else:
                fill_value = 0.0
        
        col_float[mask] = fill_value
        return col_float, fill_value

    def encode_categorical(self, column, mapping=None):
        """
        Label Encoding (Stateful).
        - Train: Tạo mapping từ dữ liệu.
        - Test: Dùng mapping có sẵn, giá trị lạ -> 0.
        """
        # 1. Train Phase: Tạo mapping mới
        if mapping is None:
            unique_values = np.unique(column)
            mapping = {val: idx for idx, val in enumerate(unique_values)}
        
        # 2. Transform Phase: Map dữ liệu
        # get(val, 0): Nếu gặp giá trị lạ chưa từng thấy, gán tạm là 0 (hoặc -1 tùy chọn)
        encoded = np.array([mapping.get(val, 0) for val in column])
        
        return encoded, mapping

    def one_hot_encode(self, col, categories=None):
        """
        One-Hot Encoding (Stateful) - Sử dụng Broadcasting.
        """
        # 1. Train Phase: Học categories
        if categories is None:
            categories = np.unique(col)
            
        mapping = {v: i for i, v in enumerate(categories)}
        
        # 2. Transform Phase
        idx_list = []
        valid_mask = []
        
        for v in col:
            if v in mapping:
                idx_list.append(mapping[v])
                valid_mask.append(True)
            else:
                # Giá trị lạ trong Test set -> Gán index 0 tạm, nhưng đánh dấu False
                idx_list.append(0)
                valid_mask.append(False)
                
        idx = np.array(idx_list)
        valid_mask = np.array(valid_mask)[:, None] # Shape (N, 1)

        # 3. Broadcasting: Tạo One-Hot Matrix
        # So sánh vector cột idx với vector hàng [0, 1, 2...]
        one_hot = (idx[:, None] == np.arange(len(categories))).astype(float)
        
        # Xóa các dòng không hợp lệ (giá trị lạ) thành toàn số 0
        one_hot = one_hot * valid_mask
        
        return one_hot, categories

    def normalize_minmax(self, column, params=None):
        """
        Min-Max Scaling (Stateful).
        """
        # 1. Train Phase: Tính min/max
        if params is None:
            col_min = np.min(column)
            col_max = np.max(column)
            params = (col_min, col_max)
        else:
            col_min, col_max = params
            
        # 2. Transform Phase: Áp dụng công thức
        if col_max - col_min == 0:
            return np.zeros(column.shape), params
            
        normalized = (column - col_min) / (col_max - col_min)
        return normalized, params

    # =========================================================================
    # 3. MAIN PIPELINES (FIT_TRANSFORM & TRANSFORM)
    # =========================================================================

    def fit_transform_features(self, data, headers, target, ordinal_features, nominal_features, numerical_features):
        """
        Pipeline xử lý dữ liệu cho TRAIN set.
        Vừa xử lý dữ liệu, vừa LƯU lại các tham số (mean, mode, mapping, min/max).
        """
        X_processed = []
        self.feature_names_processed = []
        
        # --- A. ORDINAL FEATURES ---
        # Quy trình: Fill Mode -> Label Encode -> MinMax Scale
        print("  [Train] Processing Ordinal Features...")
        for feat in ordinal_features:
            if feat in headers:
                col_idx = headers.index(feat)
                col_raw = data[:, col_idx].copy()
                
                # 1. Fill Missing (lưu mode)
                col_filled, fill_val = self.fill_missing_categorical(col_raw, fill_value=None)
                self.imputation_values[feat] = fill_val
                
                # 2. Encode (lưu mapping)
                col_encoded, mapping = self.encode_categorical(col_filled, mapping=None)
                self.ordinal_mappings[feat] = mapping
                
                # 3. Scale (lưu min/max)
                col_norm, params = self.normalize_minmax(col_encoded.astype(float), params=None)
                self.numerical_params[feat] = params
                
                X_processed.append(col_norm)
                self.feature_names_processed.append(feat)

        # --- B. NOMINAL FEATURES ---
        # Quy trình: Fill Mode -> One-Hot Encode
        print("  [Train] Processing Nominal Features...")
        for feat in nominal_features:
            if feat in headers:
                col_idx = headers.index(feat)
                col_raw = data[:, col_idx].copy()
                
                # 1. Fill Missing (lưu mode)
                col_filled, fill_val = self.fill_missing_categorical(col_raw, fill_value=None)
                self.imputation_values[feat] = fill_val
                
                # 2. One-Hot (lưu categories)
                col_oh, categories = self.one_hot_encode(col_filled, categories=None)
                self.nominal_categories_map[feat] = categories
                
                # Thêm từng cột vào kết quả
                for i in range(col_oh.shape[1]):
                    X_processed.append(col_oh[:, i])
                    self.feature_names_processed.append(f"{feat}_{categories[i]}")

        # --- C. NUMERICAL FEATURES ---
        # Quy trình: Fill Median -> MinMax Scale
        print("  [Train] Processing Numerical Features...")
        for feat in numerical_features:
            if feat in headers:
                col_idx = headers.index(feat)
                col_raw = data[:, col_idx].copy()
                
                # 1. Fill Missing (lưu median)
                col_filled, fill_val = self.fill_missing_numerical(col_raw, fill_value=None)
                self.imputation_values[feat] = fill_val
                
                # 2. Scale (lưu min/max)
                col_norm, params = self.normalize_minmax(col_filled, params=None)
                self.numerical_params[feat] = params
                
                X_processed.append(col_norm)
                self.feature_names_processed.append(feat)

        return np.column_stack(X_processed), target

    def transform_features(self, data, headers, ordinal_features, nominal_features, numerical_features):
        """
        Pipeline xử lý dữ liệu cho TEST set.
        CHỈ SỬ DỤNG lại các tham số đã học từ Train. KHÔNG học mới.
        """
        X_processed = []
        
        # --- A. ORDINAL FEATURES ---
        for feat in ordinal_features:
            if feat in headers:
                col_idx = headers.index(feat)
                col_raw = data[:, col_idx].copy()
                
                # 1. Fill Missing (Dùng mode của Train)
                col_filled, _ = self.fill_missing_categorical(col_raw, fill_value=self.imputation_values.get(feat))
                
                # 2. Encode (Dùng mapping của Train)
                col_encoded, _ = self.encode_categorical(col_filled, mapping=self.ordinal_mappings.get(feat))
                
                # 3. Scale (Dùng min/max của Train)
                col_norm, _ = self.normalize_minmax(col_encoded.astype(float), params=self.numerical_params.get(feat))
                
                X_processed.append(col_norm)

        # --- B. NOMINAL FEATURES ---
        for feat in nominal_features:
            if feat in headers:
                col_idx = headers.index(feat)
                col_raw = data[:, col_idx].copy()
                
                # 1. Fill Missing (Dùng mode của Train)
                col_filled, _ = self.fill_missing_categorical(col_raw, fill_value=self.imputation_values.get(feat))
                
                # 2. One-Hot (Dùng categories của Train)
                col_oh, _ = self.one_hot_encode(col_filled, categories=self.nominal_categories_map.get(feat))
                
                for i in range(col_oh.shape[1]):
                    X_processed.append(col_oh[:, i])

        # --- C. NUMERICAL FEATURES ---
        for feat in numerical_features:
            if feat in headers:
                col_idx = headers.index(feat)
                col_raw = data[:, col_idx].copy()
                
                # 1. Fill Missing (Dùng median của Train)
                col_filled, _ = self.fill_missing_numerical(col_raw, fill_value=self.imputation_values.get(feat))
                
                # 2. Scale (Dùng min/max của Train)
                col_norm, _ = self.normalize_minmax(col_filled, params=self.numerical_params.get(feat))
                
                X_processed.append(col_norm)

        return np.column_stack(X_processed)

    # =========================================================================
    # 4. UTILS (CORRELATION, ETC.)
    # =========================================================================
    
    def detect_outliers_iqr(self, column, multiplier=1.5):
        """
        Phát hiện outliers sử dụng phương pháp IQR (Interquartile Range)
        
        Công thức:
            IQR = Q3 - Q1
            Lower Bound = Q1 - multiplier * IQR
            Upper Bound = Q3 + multiplier * IQR
            Outliers: giá trị < Lower Bound hoặc > Upper Bound
        
        Args:
            column: numpy array hoặc list chứa dữ liệu số
            multiplier: hệ số nhân với IQR (mặc định 1.5)
                    - 1.5: phát hiện outliers "moderate" (tiêu chuẩn Tukey)
                    - 3.0: phát hiện outliers "extreme"
        
        Returns:
            dict chứa:
                - 'outlier_indices': numpy array các index của outliers
                - 'outlier_values': numpy array các giá trị outliers
                - 'n_outliers': số lượng outliers
                - 'outlier_mask': boolean mask (True = outlier)
                - 'lower_bound': ngưỡng dưới
                - 'upper_bound': ngưỡng trên
                - 'q1': quartile 1 (25%)
                - 'q3': quartile 3 (75%)
                - 'iqr': interquartile range
        
        Example:
            >>> col = np.array([1, 2, 3, 4, 5, 100])  # 100 là outlier
            >>> result = processor.detect_outliers_iqr(col, multiplier=1.5)
            >>> print(result['n_outliers'])  # 1
            >>> print(result['outlier_values'])  # [100]
        """
        # Chuyển về numpy array và loại bỏ NaN
        col = np.array(column).astype(float)
        
        # Xử lý missing values
        valid_mask = ~np.isnan(col)
        col_valid = col[valid_mask]
        
        if len(col_valid) == 0:
            return {
                'outlier_indices': np.array([]),
                'outlier_values': np.array([]),
                'n_outliers': 0,
                'outlier_mask': np.zeros(len(col), dtype=bool),
                'lower_bound': None,
                'upper_bound': None,
                'q1': None,
                'q3': None,
                'iqr': None
            }
        
        # Tính Q1, Q3, IQR
        q1 = np.percentile(col_valid, 25)
        q3 = np.percentile(col_valid, 75)
        iqr = q3 - q1
        
        # Tính ngưỡng
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        # Phát hiện outliers (chỉ xét các giá trị valid)
        outlier_mask_valid = (col_valid < lower_bound) | (col_valid > upper_bound)
        
        # Tạo mask đầy đủ cho toàn bộ column (bao gồm cả NaN)
        outlier_mask_full = np.zeros(len(col), dtype=bool)
        outlier_mask_full[valid_mask] = outlier_mask_valid
        
        # Lấy indices và values của outliers
        outlier_indices = np.where(outlier_mask_full)[0]
        outlier_values = col[outlier_mask_full]
        
        return {
            'outlier_indices': outlier_indices,
            'outlier_values': outlier_values,
            'n_outliers': len(outlier_indices),
            'outlier_mask': outlier_mask_full,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'q1': q1,
            'q3': q3,
            'iqr': iqr
        }
    
    def detect_outliers_zscore(self, column, threshold=3):
        """
        Phát hiện outliers sử dụng phương pháp Z-score (Standard Score)
        
        Công thức:
            Z-score = (x - μ) / σ
            Outliers: |Z-score| > threshold
        
        Phương pháp này giả định dữ liệu có phân phối chuẩn (Normal Distribution).
        Nếu dữ liệu không chuẩn, nên dùng IQR hoặc Modified Z-score.
        
        Args:
            column: numpy array hoặc list chứa dữ liệu số
            threshold: ngưỡng Z-score (mặc định 3)
                    - 2: ~95% dữ liệu bình thường (phát hiện nhiều outliers)
                    - 3: ~99.7% dữ liệu bình thường (tiêu chuẩn)
                    - 4: ~99.99% dữ liệu bình thường (ít false positive)
        
        Returns:
            dict chứa:
                - 'outlier_indices': numpy array các index của outliers
                - 'outlier_values': numpy array các giá trị outliers
                - 'n_outliers': số lượng outliers
                - 'outlier_mask': boolean mask (True = outlier)
                - 'z_scores': Z-score của tất cả giá trị
                - 'mean': mean của dữ liệu
                - 'std': standard deviation của dữ liệu
                - 'threshold': ngưỡng sử dụng
        
        Example:
            >>> col = np.array([1, 2, 3, 4, 5, 100])  # 100 là outlier
            >>> result = processor.detect_outliers_zscore(col, threshold=3)
            >>> print(result['n_outliers'])  # 1
            >>> print(result['outlier_values'])  # [100]
        """
        # Chuyển về numpy array và loại bỏ NaN
        col = np.array(column).astype(float)
        
        # Xử lý missing values
        valid_mask = ~np.isnan(col)
        col_valid = col[valid_mask]
        
        if len(col_valid) == 0:
            return {
                'outlier_indices': np.array([]),
                'outlier_values': np.array([]),
                'n_outliers': 0,
                'outlier_mask': np.zeros(len(col), dtype=bool),
                'z_scores': np.full(len(col), np.nan),
                'mean': None,
                'std': None,
                'threshold': threshold
            }
        
        # Tính mean và standard deviation
        mean = np.mean(col_valid)
        std = np.std(col_valid)
        
        # Xử lý trường hợp std = 0 (tất cả giá trị giống nhau)
        if std == 0:
            return {
                'outlier_indices': np.array([]),
                'outlier_values': np.array([]),
                'n_outliers': 0,
                'outlier_mask': np.zeros(len(col), dtype=bool),
                'z_scores': np.zeros(len(col)),
                'mean': mean,
                'std': std,
                'threshold': threshold
            }
        
        # Tính Z-scores cho tất cả giá trị valid
        z_scores_valid = (col_valid - mean) / std
        
        # Tạo mảng z_scores đầy đủ (bao gồm NaN)
        z_scores_full = np.full(len(col), np.nan)
        z_scores_full[valid_mask] = z_scores_valid
        
        # Phát hiện outliers: |Z-score| > threshold
        outlier_mask_valid = np.abs(z_scores_valid) > threshold
        
        # Tạo mask đầy đủ cho toàn bộ column
        outlier_mask_full = np.zeros(len(col), dtype=bool)
        outlier_mask_full[valid_mask] = outlier_mask_valid
        
        # Lấy indices và values của outliers
        outlier_indices = np.where(outlier_mask_full)[0]
        outlier_values = col[outlier_mask_full]
        
        return {
            'outlier_indices': outlier_indices,
            'outlier_values': outlier_values,
            'n_outliers': len(outlier_indices),
            'outlier_mask': outlier_mask_full,
            'z_scores': z_scores_full,
            'mean': mean,
            'std': std,
            'threshold': threshold
        }
    
    def calculate_correlation(self, col1, col2):
        """Tính Pearson Correlation Coefficient"""
        # Đảm bảo input là float
        c1 = col1.astype(float)
        c2 = col2.astype(float)
        
        mean1 = np.mean(c1)
        mean2 = np.mean(c2)
        
        numerator = np.sum((c1 - mean1) * (c2 - mean2))
        denominator = np.sqrt(np.sum((c1 - mean1)**2) * np.sum((c2 - mean2)**2))
        
        if denominator == 0: return 0
        return numerator / denominator
    
    def compute_correlation_matrix(self, data):
        """Tính ma trận correlation"""
        n_features = data.shape[1]
        corr_matrix = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(n_features):
                if i == j: corr_matrix[i, j] = 1.0
                else: corr_matrix[i, j] = self.calculate_correlation(data[:, i], data[:, j])
        return corr_matrix
    
    def get_statistics(self, column):
        """
        Tính các thống kê mô tả cho một cột
        """
        stats = {
            'mean': np.mean(column),
            'median': np.median(column),
            'std': np.std(column),
            'min': np.min(column),
            'max': np.max(column),
            'q25': np.percentile(column, 25),
            'q75': np.percentile(column, 75)
        }
        return stats