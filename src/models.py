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
    
    def fit(self, X, y, verbose=False, early_stopping=True, patience=10, min_delta=1e-4):
        """
        Huấn luyện mô hình Logistic Regression với Gradient Descent
        
        Args:
            X: features matrix (n_samples, n_features)
            y: target vector (n_samples,) - chỉ chứa 0 và 1
            verbose: in thông tin trong quá trình train
            early_stopping: dừng sớm nếu loss không giảm
            patience: số iterations chờ trước khi dừng
            min_delta: ngưỡng cải thiện tối thiểu của loss
        
        Returns:
            self: để có thể chain methods
        """
        # ============ 1. VALIDATION INPUT ============
        if X.ndim != 2:
            raise ValueError(f"X phải là 2D array, nhưng có shape {X.shape}")
        
        # Đảm bảo y là 1D array
        if y.ndim != 1:
            y = y.flatten()
        
        n_samples, n_features = X.shape
        
        # Check số lượng samples khớp
        if len(y) != n_samples:
            raise ValueError(f"X có {n_samples} samples nhưng y có {len(y)} samples")
        
        # Check y chỉ chứa 0 và 1
        unique_values = np.unique(y)
        if not (len(unique_values) <= 2 and all(v in [0, 1] for v in unique_values)):
            raise ValueError(f"y phải chỉ chứa 0 và 1, nhưng có: {unique_values}")
        
        if len(unique_values) == 1:
            print(f"⚠️  WARNING: y chỉ có 1 class duy nhất ({unique_values[0]}). Model sẽ không học được gì!")
        
        # ============ 2. KHỞI TẠO ============
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Khởi tạo weights và bias
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
        self.losses = []
        
        # Early stopping variables
        best_loss = float('inf')
        best_weights = None
        best_bias = None
        patience_counter = 0
        
        # ============ 3. GRADIENT DESCENT ============
        for i in range(self.n_iterations):
            # --- Forward pass ---
            linear_output = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_output)
            
            # --- Compute loss ---
            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)
            
            # --- Backward pass - tính gradients ---
            dz = y_pred - y
            dw = (1 / n_samples) * np.dot(X.T, dz)
            db = (1 / n_samples) * np.sum(dz)
            
            # Thêm gradient của regularization
            if self.regularization == 'l2':
                dw += (self.lambda_reg / n_samples) * self.weights
            elif self.regularization == 'l1':
                dw += (self.lambda_reg / n_samples) * np.sign(self.weights)
            
            # --- Update parameters ---
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # --- Early stopping check ---
            if early_stopping:
                if loss < best_loss - min_delta:
                    best_loss = loss
                    best_weights = self.weights.copy()
                    best_bias = self.bias
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose:
                        print(f" Early stopping at iteration {i+1}/{self.n_iterations}")
                        print(f"  Best loss: {best_loss:.4f}")
                    break
            else:
                # Không dùng early stopping thì vẫn lưu best weights
                if loss < best_loss:
                    best_loss = loss
                    best_weights = self.weights.copy()
                    best_bias = self.bias
            
            # --- Verbose output ---
            if verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.n_iterations}, Loss: {loss:.4f}")
        
        # ============ 4. RESTORE BEST WEIGHTS ============
        if best_weights is not None:
            self.weights = best_weights
            self.bias = best_bias
            if verbose:
                print(f" Training completed. Best loss: {best_loss:.4f}")
    
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
        n_features = X.shape[1]
        n_classes = len(self.classes)
        
        # Ma trận log probabilities (n_samples, n_classes)
        log_probas = np.zeros((n_samples, n_classes))
        
        # Vectorized: Tính log prior cho tất cả classes
        for class_idx, c in enumerate(self.classes):
            log_probas[:, class_idx] = np.log(self.class_priors[c])
        
        # Vectorized: Tính likelihood cho từng feature
        for feature_idx in range(n_features):
            X_feature = X[:, feature_idx]  # (n_samples,)
            
            for class_idx, c in enumerate(self.classes):
                params = self.feature_params[feature_idx][c]
                
                # Vectorized Gaussian probability cho toàn bộ samples
                mean = params['mean']
                std = params['std']
                
                # Tính cho tất cả samples cùng lúc
                exponent = -((X_feature - mean) ** 2) / (2 * std ** 2)
                coefficient = 1 / (np.sqrt(2 * np.pi) * std)
                prob = coefficient * np.exp(exponent)
                
                # Cộng log probability
                log_probas[:, class_idx] += np.log(prob + 1e-10) #Thêm epsilon
        
        # Chuyển log probabilities về probabilities
        probas = np.exp(log_probas)
        # Chuẩn hóa
        probas = probas / np.sum(probas, axis=1, keepdims=True)
        
        return probas
    
    def predict(self, X):
        """Dự đoán class"""
        probas = self.predict_proba(X)
        return self.classes[np.argmax(probas, axis=1)]

class KNearestNeighbors:
    """
    K-Nearest Neighbors với Vectorization (Nhanh hơn)
    """
    
    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
    
    def compute_distances_vectorized(self, X):
        """Tính khoảng cách Euclidean bằng ma trận (nhanh hơn vòng lặp)"""
        # Công thức: ||A - B||^2 = ||A||^2 + ||B||^2 - 2A.B^T
        # Sử dụng np.einsum thay vì X**2
        # 'ij,ij->i': sum of element-wise product along axis 1
        X_sq = np.einsum('ij,ij->i', X, X)[:, None]  # Shape: (n_test, 1)
        X_train_sq = np.einsum('ij,ij->i', self.X_train, self.X_train)  # Shape: (n_train,)
        
        # np.einsum thay vì np.dot(X, X_train.T)
        # 'ij,kj->ik': (n_test, n_features) x (n_train, n_features) -> (n_test, n_train)
        dot_product = np.einsum('ij,kj->ik', X, self.X_train)
        
        # Broadcasting: (n_test, 1) + (n_train,) + (n_test, n_train)
        dists_sq = X_sq + X_train_sq - 2 * dot_product
        return np.sqrt(np.maximum(dists_sq, 0))

    def predict_proba(self, X):
        """Dự đoán xác suất (cần cho ROC-AUC)"""
        n_test = X.shape[0]
        
        # 1. Tính khoảng cách
        dists = self.compute_distances_vectorized(X)
        
        # 2. Tìm k láng giềng gần nhất
        # argpartition giúp tìm top k nhanh hơn argsort
        k_indices = np.argpartition(dists, self.k, axis=1)[:, :self.k]
        
        # 3. Lấy nhãn của k láng giềng
        k_nearest_labels = self.y_train[k_indices]
        
        # 4. Tính xác suất Class 1 = (Số lượng Class 1) / k
        # np.mean sẽ tự động tính tỷ lệ số 1 (vì True=1, False=0)
        return np.mean(k_nearest_labels == 1, axis=1)
    
    def predict(self, X, threshold=0.5):
        """Dự đoán nhãn"""
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)


class NeuralNetwork:
    """
    Mạng Nơ-ron nhân tạo (MLP) 2 lớp được implement từ đầu bằng NumPy.
    Kiến trúc: Input -> Hidden (ReLU) -> Output (Sigmoid)
    """
    def __init__(self, input_size, hidden_size=64, learning_rate=0.1, n_iterations=1000, random_state=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.params = {}
        self.losses = []
        
    def _init_weights(self):
        """Khởi tạo trọng số ngẫu nhiên"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        # Kaiming Initialization cho lớp ReLU (giúp hội tụ nhanh hơn)
        self.params['W1'] = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2. / self.input_size)
        self.params['b1'] = np.zeros((1, self.hidden_size))
        
        # Xavier Initialization cho lớp Sigmoid
        self.params['W2'] = np.random.randn(self.hidden_size, 1) * np.sqrt(1. / self.hidden_size)
        self.params['b2'] = np.zeros((1, 1))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))

    def forward(self, X):
        """Lan truyền xuôi (Forward Propagation)"""
        # Lớp ẩn (Hidden Layer)
        Z1 = np.dot(X, self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        
        # Lớp đầu ra (Output Layer)
        Z2 = np.dot(A1, self.params['W2']) + self.params['b2']
        A2 = self.sigmoid(Z2)
        
        # Lưu cache để dùng cho backprop
        cache = (Z1, A1, Z2, A2)
        return A2, cache

    def backward(self, X, y, cache):
        """Lan truyền ngược (Backpropagation) - Tính Gradient"""
        m = X.shape[0]
        Z1, A1, Z2, A2 = cache
        y = y.reshape(-1, 1) # Đảm bảo shape (N, 1)

        # Tính đạo hàm lớp Output (Sigmoid + CrossEntropy)
        dZ2 = A2 - y
        dW2 = (1 / m) * np.dot(A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        # Tính đạo hàm lớp Hidden (ReLU)
        dA1 = np.dot(dZ2, self.params['W2'].T)
        dZ1 = dA1 * self.relu_derivative(Z1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def fit(self, X, y, verbose=False):
        if X.shape[1] != self.input_size:
            raise ValueError(f"X.shape[1]={X.shape[1]} không khớp input_size={self.input_size}")
    
        self._init_weights()
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        
        for i in range(self.n_iterations):
            # 1. Forward
            y_pred, cache = self.forward(X)
            
            # 2. Compute Loss (Binary Cross Entropy)
            epsilon = 1e-15
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
            self.losses.append(loss)
            
            # 3. Backward
            grads = self.backward(X, y, cache)
            
            # 4. Update Weights (Gradient Descent)
            self.params['W1'] -= self.learning_rate * grads['W1']
            self.params['b1'] -= self.learning_rate * grads['b1']
            self.params['W2'] -= self.learning_rate * grads['W2']
            self.params['b2'] -= self.learning_rate * grads['b2']
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Epoch {i+1}/{self.n_iterations}, Loss: {loss:.4f}")
        return self

    def predict_proba(self, X):
        A2, _ = self.forward(X)
        return A2

    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int).flatten()


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
        # Flatten nếu là 2D array
        y_true = y_true.flatten() if y_true.ndim > 1 else y_true
        y_pred = y_pred.flatten() if y_pred.ndim > 1 else y_pred
        
        # Ép kiểu về int để đảm bảo tính toán bitwise hoặc so sánh
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        
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
            return 0.0
        return tp / (tp + fp)
    
    @staticmethod
    def recall(y_true, y_pred):
        """Recall = TP / (TP + FN)"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)
    
    @staticmethod
    def f1_score(y_true, y_pred):
        """F1 = 2 * (Precision * Recall) / (Precision + Recall)"""
        prec = ModelEvaluator.precision(y_true, y_pred)
        rec = ModelEvaluator.recall(y_true, y_pred)
        
        if prec + rec == 0:
            return 0.0
        return 2 * (prec * rec) / (prec + rec)
    
    @staticmethod
    def roc_auc_score(y_true, y_pred_proba):
        """
        Tính diện tích dưới đường cong ROC (AUC) sử dụng NumPy.
        Logic: Sắp xếp sample theo score giảm dần, tính tích lũy True Positive và False Positive.
        """
        # Đảm bảo y_true là 0 và 1
        y_true = np.array(y_true)
        
        # Sắp xếp giảm dần theo xác suất dự đoán
        desc_score_indices = np.argsort(y_pred_proba)[::-1]
        y_true = y_true[desc_score_indices]
        y_score = y_pred_proba[desc_score_indices]

        # Tính số lượng Positive và Negative
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        # Tránh chia cho 0
        if n_pos == 0 or n_neg == 0:
            return 0.5

        # Tính TPR và FPR tại mọi điểm threshold (tương ứng với mỗi sample)
        # np.cumsum giúp tính tổng tích lũy số lượng True Positive và False Positive
        tpr = np.cumsum(y_true) / n_pos
        fpr = np.cumsum(1 - y_true) / n_neg

        # Chèn điểm (0,0) vào đầu để đường cong bắt đầu từ gốc tọa độ
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])

        # Tính diện tích bằng quy tắc hình thang (Trapezoidal rule)
        # Lưu ý: fpr đã được sắp xếp tăng dần tự nhiên do ta sort score giảm dần
        return np.trapz(tpr, fpr)
    
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
            elif metric == 'roc_auc':
                # Cần predict_proba cho AUC
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_val_fold)
                    # Xử lý shape output của predict_proba
                    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                        y_prob = y_prob[:, 1]
                    elif y_prob.ndim == 2 and y_prob.shape[1] == 1:
                        y_prob = y_prob.ravel()
                    score = ModelEvaluator.roc_auc_score(y_val_fold, y_prob)
                else:
                    print("Model does not support predict_proba for AUC")
                    score = 0.0
            
            scores.append(score)
        
        return np.array(scores)