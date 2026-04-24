"""
Extension methods beyond the required three (LinReg, LogReg, KNN):

    LinearSVM                       — linear multi-class SVM with hinge loss
    KernelRidgeRegression           — dual-form ridge with linear/poly/RBF kernel
    KernelLeastSquaresClassifier    — multi-output KRR on one-hot labels

These are referenced in §3.1–§3.2 of the report as comparisons to the
required parametric linear methods.
"""

from .svm import LinearSVM
from .kernel_ridge import KernelRidgeRegression, KernelLeastSquaresClassifier

__all__ = ["LinearSVM", "KernelRidgeRegression", "KernelLeastSquaresClassifier"]
