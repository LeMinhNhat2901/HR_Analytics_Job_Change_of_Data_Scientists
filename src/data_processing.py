"""
Data Processing Module - CHỈ SỬ DỤNG NUMPY
Module xử lý dữ liệu cho HR Analytics Job Change Prediction
"""

import numpy as np

class DataProcessor:
    """Class xử lý dữ liệu sử dụng NumPy"""
    
    def __init__(self):
        self.feature_names = []
        self.categorical_mappings = {}
        self.numerical_stats = {}
        
    def load_csv(self, filepath, delimiter=','):
        """
        Đọc file CSV sử dụng NumPy
        
        Args:
            filepath: đường dẫn đến file CSV
            delimiter: ký tự phân cách
            
        Returns:
            data: mảng numpy chứa dữ liệu
            headers: danh sách tên cột
        """
        # Đọc headers
        with open(filepath, 'r', encoding='utf-8') as f:
            headers = f.readline().strip().split(delimiter)
        
        # Đọc dữ liệu
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            next(f)  # Skip header
            for line in f:
                row = line.strip().split(delimiter)
                data.append(row)
        
        self.feature_names = headers
        return np.array(data, dtype=object), headers
    
    def get_column_by_name(self, data, headers, col_name):
        """Lấy cột dữ liệu theo tên"""
        idx = headers.index(col_name)
        return data[:, idx]
    
    def get_columns_by_names(self, data, headers, col_names):
        """Lấy nhiều cột theo danh sách tên"""
        indices = [headers.index(name) for name in col_names]
        return data[:, indices]
    
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
    
    def fill_missing_categorical(self, column, strategy='mode'):
        """
        Điền missing values cho biến categorical
        
        Args:
            column: cột dữ liệu
            strategy: 'mode' hoặc 'constant'
        """
        mask = np.array([
            (val == '' or val == 'nan' or val == 'NaN' or 
             val is None or str(val).lower() == 'nan')
            for val in column
        ])
        
        if strategy == 'mode':
            # Tìm giá trị xuất hiện nhiều nhất
            valid_values = column[~mask]
            if len(valid_values) > 0:
                unique, counts = np.unique(valid_values, return_counts=True)
                mode_value = unique[np.argmax(counts)]
                column[mask] = mode_value
        elif strategy == 'constant':
            column[mask] = 'Unknown'
        
        return column
    
    def fill_missing_numerical(self, column, strategy='median'):
        """
        Điền missing values cho biến numerical
        
        Args:
            column: cột dữ liệu
            strategy: 'mean', 'median', hoặc giá trị cụ thể
        """
        mask = np.array([
            (val == '' or val == 'nan' or val == 'NaN' or 
             val is None or str(val).lower() == 'nan')
            for val in column
        ])
        
        # Convert to float
        column_float = np.zeros(len(column), dtype=float)
        for i, val in enumerate(column):
            if not mask[i]:
                try:
                    column_float[i] = float(val)
                except:
                    mask[i] = True
        
        valid_values = column_float[~mask]
        
        if strategy == 'mean' and len(valid_values) > 0:
            fill_value = np.mean(valid_values)
        elif strategy == 'median' and len(valid_values) > 0:
            fill_value = np.median(valid_values)
        else:
            fill_value = 0
        
        column_float[mask] = fill_value
        return column_float
    
    def encode_categorical(self, column, create_mapping=True):
        """
        Encode categorical variable thành số
        
        Args:
            column: cột dữ liệu categorical
            create_mapping: tạo mapping mới hay dùng mapping có sẵn
            
        Returns:
            encoded: mảng đã encode
            mapping: dictionary mapping từ category sang số
        """
        unique_values = np.unique(column)
        
        if create_mapping:
            mapping = {val: idx for idx, val in enumerate(unique_values)}
        else:
            mapping = self.categorical_mappings.get(id(column), {})
        
        encoded = np.array([mapping.get(val, -1) for val in column])
        return encoded, mapping
    
    def normalize_minmax(self, column, feature_range=(0, 1)):
        """
        Min-Max normalization
        X_norm = (X - X_min) / (X_max - X_min) * (max - min) + min
        """
        col_min = np.min(column)
        col_max = np.max(column)
        
        if col_max - col_min == 0:
            return np.full(column.shape, feature_range[0])
        
        normalized = (column - col_min) / (col_max - col_min)
        normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
        
        return normalized
    
    def standardize_zscore(self, column):
        """
        Z-score standardization
        X_std = (X - mean) / std
        """
        mean = np.mean(column)
        std = np.std(column)
        
        if std == 0:
            return column - mean
        
        standardized = (column - mean) / std
        return standardized
    
    def log_transform(self, column, offset=1):
        """
        Log transformation để xử lý skewed distribution
        """
        return np.log(column + offset)
    
    def detect_outliers_iqr(self, column, multiplier=1.5):
        """
        Phát hiện outliers sử dụng IQR method
        
        Returns:
            outlier_mask: boolean array đánh dấu outliers
        """
        q1 = np.percentile(column, 25)
        q3 = np.percentile(column, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        outlier_mask = (column < lower_bound) | (column > upper_bound)
        return outlier_mask
    
    def detect_outliers_zscore(self, column, threshold=3):
        """
        Phát hiện outliers sử dụng Z-score method
        """
        mean = np.mean(column)
        std = np.std(column)
        
        if std == 0:
            return np.zeros(len(column), dtype=bool)
        
        z_scores = np.abs((column - mean) / std)
        outlier_mask = z_scores > threshold
        return outlier_mask
    
    def create_interaction_features(self, col1, col2):
        """
        Tạo interaction features giữa 2 biến
        """
        return col1 * col2
    
    def create_polynomial_features(self, column, degree=2):
        """
        Tạo polynomial features
        """
        features = [column]
        for d in range(2, degree + 1):
            features.append(column ** d)
        return np.column_stack(features)
    
    def train_test_split(self, X, y, test_size=0.2, random_state=None):
        """
        Chia dữ liệu thành train và test sets
        
        Args:
            X: features
            y: target
            test_size: tỷ lệ test set
            random_state: seed cho random
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        n_samples = X.shape[0]
        n_test = int(n_samples * test_size)
        
        # Shuffle indices
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        return X_train, X_test, y_train, y_test
    
    def calculate_correlation(self, col1, col2):
        """
        Tính correlation coefficient giữa 2 biến
        """
        mean1 = np.mean(col1)
        mean2 = np.mean(col2)
        
        numerator = np.sum((col1 - mean1) * (col2 - mean2))
        denominator = np.sqrt(np.sum((col1 - mean1)**2) * np.sum((col2 - mean2)**2))
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    def compute_correlation_matrix(self, data):
        """
        Tính ma trận correlation cho toàn bộ dữ liệu
        """
        n_features = data.shape[1]
        corr_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    corr_matrix[i, j] = self.calculate_correlation(
                        data[:, i], data[:, j]
                    )
        
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