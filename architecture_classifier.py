"""
Architecture Classification Project - Dynamic Scan & Fully Automated Version
3-Class Classification: Hanok vs Modern vs Gothic Architecture
Course: Introduction to Artificial Intelligence
"""

import os
import numpy as np
import cv2  # Official OpenCV Integration for standardized image loading
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ArchitectureClassifier:
    def __init__(self, image_dir):
        """Initialize classifier with image directory"""
        self.image_dir = Path(image_dir)
        self.models = {
            'Neural Network': MLPClassifier(hidden_layer_sizes=(500,), max_iter=500, random_state=42),
            'SVM': SVC(probability=True, random_state=42, kernel='rbf'),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        }
        self.results = {}
        self.cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def find_actual_folders(self):
        """Scan directory dynamically to find the correct 3 architecture folders (Korean or English names)"""
        if not self.image_dir.exists():
            return None, None, None
        subdirs = [d.name for d in self.image_dir.iterdir() if d.is_dir()]
        
        hanok_folder = next((d for d in subdirs if '한옥' in d or 'Hanok' in d), None)
        modern_folder = next((d for d in subdirs if '현대' in d or 'Modern' in d), None)
        gothic_folder = next((d for d in subdirs if '고딕' in d or 'Gothic' in d or 'gothic' in d), None)
        
        return hanok_folder, modern_folder, gothic_folder

    def load_images(self, category_dir):
        """Load and preprocess images from a directory using official OpenCV"""
        images = []
        labels = []
        category_path = self.image_dir / category_dir

        # Comprehensive support for multi-format expansion profiles
        image_extensions = ('*.webp', '*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG')
        for ext in image_extensions:
            for img_file in category_path.glob(ext):
                try:
                    img = cv2.imread(str(img_file))
                    if img is None:
                        continue
                    # Recover BGR to RGB space alignment safely
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, (128, 128), interpolation=cv2.INTER_LINEAR)
                    img_normalized = img_resized.astype('float32') / 255.0
                    
                    images.append(img_normalized.flatten())
                    labels.append(category_dir)
                except Exception as e:
                    pass
        return np.array(images), np.array(labels)

    def load_all_data(self):
        """Dynamically match subfolders and build the unified dataset arrays"""
        hanok_f, modern_f, gothic_f = self.find_actual_folders()
        
        if not hanok_f or not modern_f or not gothic_f:
            print("⚠️ Could not match all 3 architectural subfolders automatically.")
            print("Creating fallback synthetic framework to guarantee file execution...")
            np.random.seed(42)
            return np.random.rand(100, 128 * 128 * 3), np.random.randint(0, 3, 100), True

        print(f"📁 Auto-Detected Target Folders: Hanok->'{hanok_f}', Modern->'{modern_f}', Gothic->'{gothic_f}'")
        
        X_h, y_h = self.load_images(hanok_f)
        X_m, y_m = self.load_images(modern_f)
        X_g, y_g = self.load_images(gothic_f)

        images_list = [img for img in [X_h, X_m, X_g] if len(img) > 0]
        
        if not images_list:
            print("⚠️ Subfolders found, but they don't seem to contain supported image files.")
            print("Creating fallback synthetic framework to guarantee file generation...")
            np.random.seed(42)
            return np.random.rand(100, 128 * 128 * 3), np.random.randint(0, 3, 100), True

        X = np.vstack(images_list)
        y = np.hstack([y_h, y_m, y_g])
        
        label_map = {hanok_f: 0, modern_f: 1, gothic_f: 2}
        y_numeric = np.array([label_map[label] for label in y])
        return X, y_numeric, False

    def evaluate_models(self, X, y):
        """Perform unified cross-validation pipeline across all chosen classifiers"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for model_name, model in self.models.items():
            cv_scores = cross_val_score(model, X_scaled, y, cv=self.cv_strategy, scoring='accuracy')
            cv_f1 = cross_val_score(model, X_scaled, y, cv=self.cv_strategy, scoring='f1_macro')
            cv_roc_auc = cross_val_score(model, X_scaled, y, cv=self.cv_strategy, scoring='roc_auc_ovr')

            self.results[model_name] = {
                'accuracy_scores': cv_scores,
                'f1_scores': cv_f1,
                'roc_auc_scores': cv_roc_auc,
                'model': model
            }

    def generate_plots(self, X, y):
        """Generate, compute, and save all 3 core evaluation plots into current directory"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        class_labels = ['Hanok', 'Modern', 'Gothic']
        
        # 1. Heatmap Confusion Matrices
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for idx, (model_name, model) in enumerate(self.models.items()):
            y_pred = cross_val_predict(model, X_scaled, y, cv=self.cv_strategy)
            cm = confusion_matrix(y, y_pred, labels=[0, 1, 2])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues',
                        xticklabels=class_labels, yticklabels=class_labels)
            axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300)
        plt.close()
        print("📊 [SUCCESS] Saved 'confusion_matrices.png'")

        # 2. Multi-Class ROC Curves (OvR)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        y_bin = label_binarize(y, classes=[0, 1, 2])
        colors = ['#1f77b4', '#d62728', '#2ca02c']
        for idx, (model_name, model) in enumerate(self.models.items()):
            ax = axes[idx]
            y_proba = cross_val_predict(model, X_scaled, y, cv=self.cv_strategy, method='predict_proba')
            for i, color in zip(range(3), colors):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{class_labels[i]} (AUC = {auc(fpr, tpr):.3f})')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
            ax.set_title(f'{model_name} ROC Curve', fontsize=12, fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300)
        plt.close()
        print("📊 [SUCCESS] Saved 'roc_curves.png'")

        # 3. Bar Chart Performance Benchmarking
        model_names = list(self.results.keys())
        acc_means = [self.results[m]['accuracy_scores'].mean() for m in model_names]
        f1_means = [self.results[m]['f1_scores'].mean() for m in model_names]
        auc_means = [self.results[m]['roc_auc_scores'].mean() for m in model_names]

        x = np.arange(len(model_names))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width, acc_means, width, label='Accuracy', color='#4c72b0')
        ax.bar(x, f1_means, width, label='F1 Score (Macro)', color='#dd8452')
        ax.bar(x + width, auc_means, width, label='ROC-AUC (OvR)', color='#55a868')
        ax.set_ylabel('Scores')
        ax.set_title('Model Performance Comparison (5-Fold Stratified CV)', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend(loc='lower right')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300)
        plt.close()
        print("📊 [SUCCESS] Saved 'performance_comparison.png'")

    def write_report_file(self, X, y, is_fallback):
        """Generate evaluation_report.txt logging all statistical summary details"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        class_names = ['Hanok (한옥)', 'Modern (현대)', 'Gothic']
        
        with open('evaluation_report.txt', 'w', encoding='utf-8') as f:
            f.write("="*75 + "\n")
            f.write("       ARCHITECTURE CLASSIFICATION PROJECT - PERFORMANCE REPORT\n")
            f.write("             Course: Introduction to Artificial Intelligence\n")
            f.write("="*75 + "\n\n")
            f.write("1. DATASET STATISTICS\n")
            status_text = "Generated Mock Dataset Framework" if is_fallback else "Real User Image Dataset Loaded"
            f.write(f"  • Data Loading Mode: {status_text}\n")
            f.write(f"  • Total Processed Images: {len(y)} samples\n\n")
            f.write("2. 5-FOLD STRATIFIED CROSS-VALIDATION METRICS\n")
            f.write("-" * 50 + "\n")
            for model_name in self.results:
                res = self.results[model_name]
                f.write(f"\n▶ Algorithm Model: {model_name}\n")
                f.write(f"   Mean Accuracy : {res['accuracy_scores'].mean():.4f}\n")
                f.write(f"   Mean F1 Macro : {res['f1_scores'].mean():.4f}\n")
                f.write(f"   Mean ROC-AUC  : {res['roc_auc_scores'].mean():.4f}\n")
                y_pred = cross_val_predict(res['model'], X_scaled, y, cv=self.cv_strategy)
                report = classification_report(y, y_pred, target_names=class_names, output_dict=True, zero_division=0)
                f.write(f"   Detailed Metrics Profile:\n")
                for c_name in class_names:
                    m = report[c_name]
                    f.write(f"     - {c_name:15s} -> Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | F1: {m['f1-score']:.4f}\n")
        print("📝 [SUCCESS] Official classification text report saved to 'evaluation_report.txt'.")

def main():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    if current_directory == "":
        current_directory = os.getcwd()
    
    arch_target_path = Path(current_directory) / "arch"
    if not arch_target_path.exists():
        arch_target_path = Path(current_directory)
        
    classifier = ArchitectureClassifier(arch_target_path)
    X, y, is_fallback = classifier.load_all_data()
    
    classifier.evaluate_models(X, y)
    classifier.generate_plots(X, y)
    classifier.write_report_file(X, y, is_fallback)

if __name__ == "__main__":
    main()
