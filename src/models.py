"""
Machine Learning Models Module - CHỈ SỬ DỤNG NUMPY
Các mô hình học máy được implement từ đầu bằng NumPy
"""

import numpy as np

class LogisticRegression:
    """
    Logistic Regression được implement từ đầu bằng NumPy
    Sử dụng Gradient Descent để tối ưu
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, 
                 regularization='l2', lambda_reg=0.01, random_state=None):
        """
        Args:
            learning_rate: tốc độ học
            n_iterations: số lần lặp
            regularization: 'l1', 'l2', hoặc None
            lambda_reg: hệ số regularization
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.losses = []
    
    def sigmoid(self, z):
        """Hàm sigmoid: σ(z) = 1 / (1 + e^(-z))"""
        # Clip để tránh overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, y_true, y_pred):
        """
        Binary Cross-Entropy Loss
        L = -1/m * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
        """
        m = len(y_true)
        epsilon = 1e-15  # Tránh log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -np.mean(y_true * np.log(y_pred) + 
                       (1 - y_true) * np.log(1 - y_pred))
        
        # Thêm regularization
        if self.regularization == 'l2':
            loss += (self.lambda_reg / (2 * m)) * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            loss += (self.lambda_reg / m) * np.sum(np.abs(self.weights))
        
        return loss
    
    def fit(self, X, y, verbose=False):
        """
        Huấn luyện mô hình
        
        Args:
            X: features matrix (n_samples, n_features)
            y: target vector (n_samples,)
            verbose: in thông tin trong quá trình train
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # Khởi tạo weights và bias
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
        
        # Gradient Descent
        for i in range(self.n_iterations):
            # Forward pass
            linear_output = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_output)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)
            
            # Backward pass - tính gradients
            dz = y_pred - y
            dw = (1 / n_samples) * np.dot(X.T, dz)
            db = (1 / n_samples) * np.sum(dz)
            
            # Thêm gradient của regularization
            if self.regularization == 'l2':
                dw += (self.lambda_reg / n_samples) * self.weights
            elif self.regularization == 'l1':
                dw += (self.lambda_reg / n_samples) * np.sign(self.weights)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.n_iterations}, Loss: {loss:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """Dự đoán xác suất"""
        linear_output = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_output)
    
    def predict(self, X, threshold=0.5):
        """Dự đoán class (0 hoặc 1)"""
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)


class NaiveBayes:
    """
    Naive Bayes Classifier được implement từ đầu bằng NumPy
    Hỗ trợ cả Gaussian và Categorical features
    """
    
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.feature_params = {}
    
    def fit(self, X, y, feature_types='auto'):
        """
        Huấn luyện mô hình
        
        Args:
            X: features matrix
            y: target vector
            feature_types: 'gaussian', 'categorical', hoặc 'auto'
        """
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        # Tính prior probabilities: P(y)
        for c in self.classes:
            self.class_priors[c] = np.sum(y == c) / n_samples
        
        # Tính likelihood parameters cho mỗi feature
        for feature_idx in range(n_features):
            self.feature_params[feature_idx] = {}
            
            for c in self.classes:
                X_c = X[y == c, feature_idx]
                
                # Gaussian: tính mean và std
                self.feature_params[feature_idx][c] = {
                    'mean': np.mean(X_c),
                    'std': np.std(X_c) + 1e-6  # Thêm epsilon tránh chia 0
                }
        
        return self
    
    def gaussian_probability(self, x, mean, std):
        """
        Tính Gaussian probability density
        P(x|μ,σ) = (1/√(2πσ²)) * e^(-(x-μ)²/(2σ²))
        """
        exponent = -((x - mean) ** 2) / (2 * std ** 2)
        coefficient = 1 / (np.sqrt(2 * np.pi) * std)
        return coefficient * np.exp(exponent)
    
    def predict_proba(self, X):
        """Dự đoán xác suất cho mỗi class"""
        n_samples = X.shape[0]
        probas = np.zeros((n_samples, len(self.classes)))
        
        for sample_idx in range(n_samples):
            for class_idx, c in enumerate(self.classes):
                # Bắt đầu với prior probability
                log_prob = np.log(self.class_priors[c])
                
                # Nhân với likelihood của mỗi feature (sử dụng log để tránh underflow)
                for feature_idx in range(X.shape[1]):
                    x_val = X[sample_idx, feature_idx]
                    params = self.feature_params[feature_idx][c]
                    
                    prob = self.gaussian_probability(
                        x_val, params['mean'], params['std']
                    )
                    log_prob += np.log(prob + 1e-10)  # Thêm epsilon
                
                probas[sample_idx, class_idx] = log_prob
        
        # Convert log probabilities back to probabilities
        probas = np.exp(probas)
        # Normalize
        probas = probas / np.sum(probas, axis=1, keepdims=True)
        
        return probas
    
    def predict(self, X):
        """Dự đoán class"""
        probas = self.predict_proba(X)
        return self.classes[np.argmax(probas, axis=1)]


class KNearestNeighbors:
    """
    K-Nearest Neighbors được implement từ đầu bằng NumPy
    """
    
    def __init__(self, k=5, metric='euclidean'):
        """
        Args:
            k: số lượng neighbors
            metric: 'euclidean' hoặc 'manhattan'
        """
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Lưu training data"""
        self.X_train = X
        self.y_train = y
        return self
    
    def euclidean_distance(self, x1, x2):
        """Khoảng cách Euclidean: √(Σ(x1 - x2)²)"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def manhattan_distance(self, x1, x2):
        """Khoảng cách Manhattan: Σ|x1 - x2|"""
        return np.sum(np.abs(x1 - x2))
    
    def predict(self, X):
        """Dự đoán cho dữ liệu mới"""
        predictions = np.zeros(X.shape[0])
        
        for i, x in enumerate(X):
            # Tính khoảng cách đến tất cả training samples
            distances = np.zeros(len(self.X_train))
            
            for j, x_train in enumerate(self.X_train):
                if self.metric == 'euclidean':
                    distances[j] = self.euclidean_distance(x, x_train)
                else:
                    distances[j] = self.manhattan_distance(x, x_train)
            
            # Lấy k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            # Voting: lấy class xuất hiện nhiều nhất
            unique, counts = np.unique(k_nearest_labels, return_counts=True)
            predictions[i] = unique[np.argmax(counts)]
        
        return predictions


class ModelEvaluator:
    """Class đánh giá hiệu suất mô hình"""
    
    @staticmethod
    def accuracy(y_true, y_pred):
        """Accuracy = (TP + TN) / Total"""
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def confusion_matrix(y_true, y_pred):
        """
        Tính confusion matrix
        Returns: [[TN, FP], [FN, TP]]
        """
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        return np.array([[tn, fp], [fn, tp]])
    
    @staticmethod
    def precision(y_true, y_pred):
        """Precision = TP / (TP + FP)"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        
        if tp + fp == 0:
            return 0
        return tp / (tp + fp)
    
    @staticmethod
    def recall(y_true, y_pred):
        """Recall = TP / (TP + FN)"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        if tp + fn == 0:
            return 0
        return tp / (tp + fn)
    
    @staticmethod
    def f1_score(y_true, y_pred):
        """F1 = 2 * (Precision * Recall) / (Precision + Recall)"""
        prec = ModelEvaluator.precision(y_true, y_pred)
        rec = ModelEvaluator.recall(y_true, y_pred)
        
        if prec + rec == 0:
            return 0
        return 2 * (prec * rec) / (prec + rec)
    
    @staticmethod
    def roc_auc_score(y_true, y_pred_proba):
        """
        Tính ROC AUC score
        Sử dụng trapezoidal rule
        """
        # Sắp xếp theo xác suất dự đoán
        sorted_indices = np.argsort(y_pred_proba)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        # Tính TPR và FPR
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        
        if n_pos == 0 or n_neg == 0:
            return 0.5
        
        tpr = np.cumsum(y_true_sorted) / n_pos
        fpr = np.cumsum(1 - y_true_sorted) / n_neg
        
        # Thêm điểm (0,0)
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])
        
        # Tính AUC bằng trapezoidal rule
        auc = np.trapz(tpr, fpr)
        
        return auc
    
    @staticmethod
    def cross_validate(model, X, y, k_folds=5, metric='accuracy'):
        """
        K-Fold Cross Validation
        
        Args:
            model: mô hình cần đánh giá
            X: features
            y: target
            k_folds: số folds
            metric: 'accuracy', 'f1', 'precision', 'recall'
            
        Returns:
            scores: list các scores cho mỗi fold
        """
        n_samples = X.shape[0]
        fold_size = n_samples // k_folds
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        scores = []
        
        for fold in range(k_folds):
            # Tạo validation fold
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < k_folds - 1 else n_samples
            val_indices = indices[val_start:val_end]
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
            
            # Split data
            X_train_fold = X[train_indices]
            y_train_fold = y[train_indices]
            X_val_fold = X[val_indices]
            y_val_fold = y[val_indices]
            
            # Train và predict
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            
            # Tính score
            if metric == 'accuracy':
                score = ModelEvaluator.accuracy(y_val_fold, y_pred)
            elif metric == 'f1':
                score = ModelEvaluator.f1_score(y_val_fold, y_pred)
            elif metric == 'precision':
                score = ModelEvaluator.precision(y_val_fold, y_pred)
            elif metric == 'recall':
                score = ModelEvaluator.recall(y_val_fold, y_pred)
            else:
                score = ModelEvaluator.accuracy(y_val_fold, y_pred)
            
            scores.append(score)
        
        return np.array(scores)