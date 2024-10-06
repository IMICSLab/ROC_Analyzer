# -*- coding: utf-8 -*-
"""
Created in Apr 2024

@author: Ernest
"""

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Set global font sizes
plt.rcParams['axes.labelsize'] = 12  # Sets the font size for X and Y labels
plt.rcParams['axes.titlesize'] = 14  # Sets the font size for the title
plt.rcParams['xtick.labelsize'] = 12 # Sets the font size for X-axis tick labels
plt.rcParams['ytick.labelsize'] = 12 # Sets the font size for Y-axis tick labels
plt.rcParams['legend.fontsize'] = 12 # Sets the font size for legend


# Ground truth labels
GT_labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# Predicted probabilities
Predicted_probs = [0.1, 0.15, 0.5, 0.6, 0.3, 0.55, 0.7, 0.8, 0.85, 0.99]


def is_point_on_roc(fpr, tpr, point):
    # Convert FPR and TPR values to a list of tuples representing points on the ROC curve
    roc_points = list(zip(fpr, tpr))

    # Check if the point exists in the list of ROC curve points
    if point in roc_points:
        return True, roc_points.index(point)
    else:
        return False, None


def IMICS_ROC_plot(gt_labels, preds, j_point=False, b_point=False):
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(gt_labels, preds)
    # print(fpr, "*******", tpr)

    # Calculate the AUC (Area under the ROC Curve)
    roc_auc = auc(fpr, tpr)

    # Calculating the number of actual negatives and positives from the GT_labels
    actual_negatives = GT_labels.count(0)
    actual_positives = GT_labels.count(1)

    # Calculating the tick intervals for the x-axis and y-axis based on the actual negatives and positives
    x_ticks = [i / actual_negatives for i in range(actual_negatives + 1)]
    y_ticks = [i / actual_positives for i in range(actual_positives + 1)]

    # Formatting tick labels to ensure uniform precision
    x_tick_labels = ['{:.3f}'.format(tick) for tick in x_ticks]
    y_tick_labels = ['{:.3f}'.format(tick) for tick in y_ticks]

    # Adjusting the plot with the new tick settings
    plt.figure(figsize=(12, 12), facecolor='white')
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='lightgray', lw=2, linestyle='--')
    if b_point:
        # Generate the label B line points based on the provided formula
        label_b_line_points = [(0 + i * 1/actual_negatives, 1 - i * (1/actual_positives)) for i in range(actual_negatives + 1)]
        # print(label_b_line_points)
        # Check each point on the adjusted line to see if it is on the ROC curve
        points_on_roc = [point for point in label_b_line_points if is_point_on_roc(fpr, tpr, point)[0]]
        if len(points_on_roc) > 1:
            print("Error! B point is not unique!")
        else:
            B_threshold = thresholds[is_point_on_roc(fpr, tpr, points_on_roc[0])[1]]
        # print(points_on_roc)
        plt.plot([0, 1], [1, 1-actual_negatives/actual_positives], color='lightgray', lw=2, linestyle='--')
        for point in points_on_roc:
            plt.scatter(point[0], point[1], color='blue', s=50, label="B point (FPR: {:.3f}, TPR: {:.3f}, threshold: {:.3f})".format(point[0], point[1], B_threshold), zorder=5)

    if j_point:
        # Calculating Youden's J statistic for each threshold
        J = tpr - fpr
        optimal_idx = J.argmax()
        optimal_threshold = thresholds[optimal_idx]
        optimal_fpr, optimal_tpr = fpr[optimal_idx], tpr[optimal_idx]
        # Plotting Youden's J point
        plt.scatter(optimal_fpr, optimal_tpr, color='red', s=100, marker = 'x', label="Youden's J point (FPR: {:.3f}, TPR: {:.3f}, threshold: {:.3f})".format(optimal_fpr, optimal_tpr,optimal_threshold), zorder=5)


    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)

    # Applying the custom tick labels with uniform precision
    plt.xticks(x_ticks, x_tick_labels)
    plt.yticks(y_ticks, y_tick_labels)


    plt.legend(loc="lower right", frameon=False)

    plt.grid(color='gray', linestyle='-', linewidth=0.5, which='both', axis='both')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    plt.show()

IMICS_ROC_plot(GT_labels, Predicted_probs, j_point=True, b_point=True)
