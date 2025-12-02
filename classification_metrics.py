#!/usr/bin/env python3
"""
Classification Metrics Calculator for Differential Abundance Analysis
Calculates confusion matrix, precision, recall, F1-score, AUC-ROC, etc.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)
import os


def load_data(results_file, labels_file):
    """Load results and ground truth labels"""
    # Load method results
    results = pd.read_csv(results_file)
    
    # Load true labels
    labels = pd.read_csv(labels_file, index_col=0)
    
    return results, labels


def calculate_metrics(y_true, y_pred, y_score=None):
    """Calculate classification metrics"""
    metrics = {}
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics['TP'] = int(tp)
    metrics['TN'] = int(tn)
    metrics['FP'] = int(fp)
    metrics['FN'] = int(fn)
    
    # Basic metrics
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred)
    metrics['Recall'] = recall_score(y_true, y_pred)
    metrics['Sensitivity'] = metrics['Recall']  # Same as recall
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['F1_Score'] = f1_score(y_true, y_pred)
    
    # False positive rate and False discovery rate
    metrics['FPR'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['FDR'] = fp / (fp + tp) if (fp + tp) > 0 else 0
    
    # AUC-ROC and AUC-PR (if scores are provided)
    if y_score is not None:
        try:
            metrics['AUC_ROC'] = roc_auc_score(y_true, y_score)
            metrics['AUC_PR'] = average_precision_score(y_true, y_score)
        except:
            metrics['AUC_ROC'] = np.nan
            metrics['AUC_PR'] = np.nan
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Calculate classification metrics for DAA results'
    )
    
    # Required arguments - use names matching config.cfg
    parser.add_argument('--results', required=True, dest='results',
                        help='Path to method results CSV file')
    parser.add_argument('--data.true_labels_proteins', required=True, dest='labels',
                        help='Path to true labels CSV file')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory')
    parser.add_argument('--name', required=True,
                        help='Dataset/method name')
    
    # Optional arguments
    parser.add_argument('--fdr-threshold', type=float, default=0.05,
                        help='FDR threshold for significance (default: 0.05)')
    parser.add_argument('--score-column', default='P.Value',
                        help='Column name for probability scores (default: P.Value)')
    parser.add_argument('--prediction-column', default='Significant',
                        help='Column name for binary predictions (default: Significant)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data...")
    print(f"  Results: {args.results}")
    print(f"  Labels: {args.labels}")
    
    # Load data
    results, labels = load_data(args.results, args.labels)
    
    print(f"\nData loaded:")
    print(f"  Results shape: {results.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    # Align data (ensure same order)
    # Assuming results have a protein ID column that matches labels index
    id_col = None
    for col in ['Name', 'ID', 'Protein', 'protein', 'Feature']:
        if col in results.columns:
            id_col = col
            break
    
    if id_col:
        results = results.set_index(id_col)
    else:
        print(f"Warning: No recognized ID column found in results. Using existing index.")
    
    # Get common features
    common_features = results.index.intersection(labels.index)
    print(f"  Common features: {len(common_features)}")
    
    results = results.loc[common_features]
    labels = labels.loc[common_features]
    
    # Extract true labels
    # Support both 'label' and 'is_differentially_expressed' column names
    if 'label' in labels.columns:
        y_true = labels['label'].values
    elif 'is_differentially_expressed' in labels.columns:
        y_true = labels['is_differentially_expressed'].values
    else:
        y_true = labels.iloc[:, 0].values  # Use first column if neither exists
    
    # Extract predictions
    if args.prediction_column in results.columns:
        y_pred = results[args.prediction_column].astype(int).values
    else:
        # Auto-detect p-value column
        p_val_col = None
        for col in ['P.Value', 'p_value', 'pval', 'p-value', 'adj.P.Val']:
            if col in results.columns:
                p_val_col = col
                break
        
        if p_val_col:
            y_pred = (results[p_val_col] < args.fdr_threshold).astype(int).values
        elif args.score_column in results.columns:
            y_pred = (results[args.score_column] < args.fdr_threshold).astype(int).values
        else:
            raise ValueError(f"Could not find p-value column. Checked: P.Value, p_value, pval, p-value, adj.P.Val, {args.score_column}")
    
    # Extract probability scores (use 1 - p-value as score, higher = more likely DE)
    y_score = None
    # Auto-detect p-value column for scoring
    p_val_col = None
    for col in ['P.Value', 'p_value', 'pval', 'p-value', 'adj.P.Val']:
        if col in results.columns:
            p_val_col = col
            break
    
    if p_val_col:
        # For p-values, we need to invert them (lower p-value = higher confidence)
        y_score = 1 - results[p_val_col].values
    elif args.score_column in results.columns:
        # For p-values, we need to invert them (lower p-value = higher confidence)
        y_score = 1 - results[args.score_column].values
    
    print(f"\nPredictions:")
    print(f"  True positives (ground truth): {y_true.sum()}")
    print(f"  Predicted positives: {y_pred.sum()}")
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_score)
    
    # Print metrics
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION METRICS")
    print(f"{'='*60}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['TP']:6d}  |  FN: {metrics['FN']:6d}")
    print(f"  FP: {metrics['FP']:6d}  |  TN: {metrics['TN']:6d}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:    {metrics['Accuracy']:.4f}")
    print(f"  Precision:   {metrics['Precision']:.4f}")
    print(f"  Recall:      {metrics['Recall']:.4f}")
    print(f"  Sensitivity: {metrics['Sensitivity']:.4f}")
    print(f"  Specificity: {metrics['Specificity']:.4f}")
    print(f"  F1-Score:    {metrics['F1_Score']:.4f}")
    print(f"  FPR:         {metrics['FPR']:.4f}")
    print(f"  FDR:         {metrics['FDR']:.4f}")
    
    if 'AUC_ROC' in metrics and not np.isnan(metrics['AUC_ROC']):
        print(f"\nAUC Metrics:")
        print(f"  AUC-ROC:     {metrics['AUC_ROC']:.4f}")
        print(f"  AUC-PR:      {metrics['AUC_PR']:.4f}")
    
    print(f"\n{'='*60}")
    
    # Save metrics to file
    metrics_file = os.path.join(args.output_dir, f"{args.name}_metrics.csv")
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_file, index=False)
    print(f"\nMetrics saved to: {metrics_file}")
    
    # Save detailed results with predictions and truth
    detailed_file = os.path.join(args.output_dir, f"{args.name}_detailed.csv")
    detailed_df = results.copy()
    detailed_df['True_Label'] = y_true
    detailed_df['Predicted_Label'] = y_pred
    if y_score is not None:
        detailed_df['Prediction_Score'] = y_score
    detailed_df.to_csv(detailed_file)
    print(f"Detailed results saved to: {detailed_file}")


if __name__ == "__main__":
    main()
