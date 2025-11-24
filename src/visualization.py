"""
Visualization Module - SỬ DỤNG MATPLOTLIB VÀ SEABORN
Module trực quan hóa dữ liệu cho HR Analytics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

class DataVisualizer:
    """Class trực quan hóa dữ liệu"""
    
    def __init__(self, style='whitegrid'):
        """Khởi tạo visualizer với style"""
        sns.set_style(style)
        self.colors = sns.color_palette("husl", 10)
    
    def plot_missing_values(self, missing_counts, feature_names, figsize=(12, 6)):
        """
        Vẽ biểu đồ missing values
        
        Args:
            missing_counts: số lượng missing values mỗi feature
            feature_names: tên các features
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, missing_counts, color='coral')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Số lượng Missing Values')
        ax.set_title('Missing Values theo Feature', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_target_distribution(self, target, figsize=(10, 5)):
        """
        Vẽ phân phối của target variable
        
        Args:
            target: mảng target values (0/1)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Count plot
        unique, counts = np.unique(target, return_counts=True)
        labels = ['Không thay đổi (0)', 'Thay đổi công việc (1)']
        ax1.bar(range(len(unique)), counts, color=['#3498db', '#e74c3c'])
        ax1.set_xticks(range(len(unique)))
        ax1.set_xticklabels(labels, ha='center')
        ax1.set_ylabel('Số lượng')
        ax1.set_title('Phân phối Target Variable', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Pie chart
        percentages = counts / np.sum(counts) * 100
        colors = ['#3498db', '#e74c3c']
        ax2.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Tỷ lệ phần trăm Target', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_categorical_distribution(self, data, feature_name, target=None, 
                                     figsize=(12, 6), top_n=10):
        """
        Vẽ phân phối của categorical variable
        
        Args:
            data: mảng dữ liệu categorical
            feature_name: tên feature
            target: mảng target (nếu muốn xem phân phối theo target)
            top_n: hiển thị top N categories
        """
        unique, counts = np.unique(data, return_counts=True)
        
        # Sắp xếp và lấy top N
        sorted_indices = np.argsort(counts)[::-1][:top_n]
        unique = unique[sorted_indices]
        counts = counts[sorted_indices]
        
        if target is None:
            fig, ax = plt.subplots(figsize=figsize)
            ax.bar(range(len(unique)), counts, color=self.colors[0])
            ax.set_xticks(range(len(unique)))
            ax.set_xticklabels(unique, ha='center')
            ax.set_ylabel('Số lượng')
            ax.set_title(f'Phân phối {feature_name}', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        else:
            # Stacked bar chart với target
            fig, ax = plt.subplots(figsize=figsize)
            
            target_0_counts = []
            target_1_counts = []
            
            for cat in unique:
                mask = data == cat
                target_0_counts.append(np.sum((mask) & (target == 0)))
                target_1_counts.append(np.sum((mask) & (target == 1)))
            
            x = np.arange(len(unique))
            width = 0.6
            
            ax.bar(x, target_0_counts, width, label='Không thay đổi', 
                   color='#3498db', alpha=0.8)
            ax.bar(x, target_1_counts, width, bottom=target_0_counts,
                   label='Thay đổi công việc', color='#e74c3c', alpha=0.8)
            
            ax.set_xticks(x)
            ax.set_xticklabels(unique, ha='center')
            ax.set_ylabel('Số lượng')
            ax.set_title(f'Phân phối {feature_name} theo Target', 
                        fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_numerical_distribution(self, data, feature_name, bins=30, 
                                   figsize=(12, 5)):
        """
        Vẽ phân phối của numerical variable
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        ax1.hist(data, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel(feature_name)
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Histogram - {feature_name}', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Box plot
        ax2.boxplot(data, vert=True)
        ax2.set_ylabel(feature_name)
        ax2.set_title(f'Box Plot - {feature_name}', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self, corr_matrix, feature_names, 
                                figsize=(12, 10)):
        """
        Vẽ heatmap correlation matrix
        
        Args:
            corr_matrix: ma trận correlation
            feature_names: tên các features
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sử dụng seaborn để vẽ heatmap đẹp hơn
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1,
                   xticklabels=feature_names, yticklabels=feature_names,
                   cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title('Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(ha='center')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        return fig
    
    def plot_scatter(self, x, y, x_label, y_label, hue=None, 
                    figsize=(10, 6)):
        """
        Vẽ scatter plot
        
        Args:
            x, y: dữ liệu trục x và y
            x_label, y_label: nhãn trục
            hue: biến để phân màu
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if hue is None:
            ax.scatter(x, y, alpha=0.6, c='steelblue', edgecolors='black', s=50)
        else:
            # Vẽ theo nhóm
            unique_hue = np.unique(hue)
            colors = self.colors[:len(unique_hue)]
            
            for i, val in enumerate(unique_hue):
                mask = hue == val
                label = 'Thay đổi' if val == 1 else 'Không thay đổi'
                ax.scatter(x[mask], y[mask], alpha=0.6, c=[colors[i]], 
                          label=label, edgecolors='black', s=50)
            ax.legend()
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'{x_label} vs {y_label}', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_training_history(self, train_losses, val_losses=None, 
                             metric_name='Loss', figsize=(10, 6)):
        """
        Vẽ biểu đồ training history
        
        Args:
            train_losses: losses trên tập train
            val_losses: losses trên tập validation (nếu có)
            metric_name: tên metric
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        epochs = np.arange(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, marker='o', linestyle='-', 
               linewidth=2, markersize=6, label=f'Training {metric_name}',
               color='#3498db')
        
        if val_losses is not None:
            ax.plot(epochs, val_losses, marker='s', linestyle='--',
                   linewidth=2, markersize=6, label=f'Validation {metric_name}',
                   color='#e74c3c')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'Training History - {metric_name}', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, figsize=(8, 6)):
        """
        Vẽ confusion matrix
        
        Args:
            y_true: nhãn thực tế
            y_pred: nhãn dự đoán
        """
        # Tính confusion matrix
        classes = np.unique(y_true)
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        
        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Vẽ heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Không thay đổi', 'Thay đổi'],
                   yticklabels=['Không thay đổi', 'Thay đổi'],
                   cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, importances, feature_names, 
                           top_n=15, figsize=(10, 8)):
        """
        Vẽ biểu đồ feature importance
        
        Args:
            importances: độ quan trọng của features
            feature_names: tên các features
            top_n: hiển thị top N features
        """
        # Ensure importances and feature_names have same length
        min_len = min(len(importances), len(feature_names))
        importances = importances[:min_len]
        feature_names = feature_names[:min_len]
        
        # Sắp xếp và lấy top N
        top_n = min(top_n, len(importances))
        sorted_indices = np.argsort(importances)[::-1][:top_n]
        sorted_importances = importances[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        y_pos = np.arange(len(sorted_names))
        ax.barh(y_pos, sorted_importances, color='steelblue', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=9)  # Reduce font size
        ax.set_xlabel('Importance (|Coefficient|)', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importances', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        
        # Use subplots_adjust instead of tight_layout
        plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.08)
        
        return fig
    
    def plot_roc_curve(self, y_true, y_pred_proba, figsize=(8, 6)):
        """
        Vẽ ROC curve
        
        Args:
            y_true: nhãn thực tế
            y_pred_proba: xác suất dự đoán
        """
        # Tính TPR và FPR cho các threshold khác nhau
        thresholds = np.linspace(0, 1, 100)
        tpr_list = []
        fpr_list = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        # Tính AUC sử dụng trapezoidal rule
        fpr_array = np.array(fpr_list)
        tpr_array = np.array(tpr_list)
        sorted_indices = np.argsort(fpr_array)
        fpr_sorted = fpr_array[sorted_indices]
        tpr_sorted = tpr_array[sorted_indices]
        
        auc = np.trapz(tpr_sorted, fpr_sorted)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(fpr_list, tpr_list, color='#e74c3c', linewidth=2,
               label=f'ROC curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_numerical_by_target(self, data, feature_name, target, figsize=(14, 6), plot_type='violin'):
        """
        Vẽ phân phối của numerical variable theo target (binary)
        
        Args:
            data: mảng dữ liệu numerical
            feature_name: tên feature
            target: mảng target (0/1)
            plot_type: loại biểu đồ ('violin', 'box', 'both')
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Chuẩn bị dữ liệu cho 2 nhóm
        data_0 = data[target == 0]
        data_1 = data[target == 1]
        
        # ===== Violin Plot (hoặc Box Plot) =====
        ax1 = axes[0]
        
        if plot_type in ['violin', 'both']:
            # Tạo positions cho violin plot
            positions = [1, 2]
            parts = ax1.violinplot([data_0, data_1], positions=positions,
                                showmeans=True, showmedians=True,
                                widths=0.7)
            
            # Tùy chỉnh màu sắc
            colors = ['#3498db', '#e74c3c']
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)
            
            # Tùy chỉnh các thành phần khác
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
                if partname in parts:
                    vp = parts[partname]
                    vp.set_edgecolor('black')
                    vp.set_linewidth(1.5)
            
            ax1.set_xticks(positions)
            ax1.set_xticklabels(['Không thay đổi (0)', 'Thay đổi công việc (1)'])
            ax1.set_ylabel(feature_name, fontsize=12)
            ax1.set_title(f'Violin Plot - {feature_name} theo Target', 
                        fontsize=13, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
            
        elif plot_type == 'box':
            # Box plot đơn giản
            bp = ax1.boxplot([data_0, data_1], positions=[1, 2],
                            widths=0.6, patch_artist=True,
                            labels=['Không thay đổi (0)', 'Thay đổi công việc (1)'])
            
            colors = ['#3498db', '#e74c3c']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax1.set_ylabel(feature_name, fontsize=12)
            ax1.set_title(f'Box Plot - {feature_name} theo Target', 
                        fontsize=13, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
        
        # ===== Histogram chồng lấp =====
        ax2 = axes[1]
        
        # Tính range chung để có bins nhất quán
        data_min = min(data_0.min(), data_1.min())
        data_max = max(data_0.max(), data_1.max())
        bins = np.linspace(data_min, data_max, 30)
        
        # Vẽ histogram
        ax2.hist(data_0, bins=bins, alpha=0.6, color='#3498db', 
                label='Không thay đổi (0)', edgecolor='black', linewidth=0.5)
        ax2.hist(data_1, bins=bins, alpha=0.6, color='#e74c3c',
                label='Thay đổi công việc (1)', edgecolor='black', linewidth=0.5)
        
        ax2.set_xlabel(feature_name, fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Histogram - {feature_name} theo Target', 
                    fontsize=13, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(axis='y', alpha=0.3)
        
        # Thêm thống kê tóm tắt
        stats_text = (
            f"Nhóm 0: μ={data_0.mean():.2f}, σ={data_0.std():.2f}\n"
            f"Nhóm 1: μ={data_1.mean():.2f}, σ={data_1.std():.2f}"
        )
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        
        return fig