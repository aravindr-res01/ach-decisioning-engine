package com.ach.research.model;

/**
 * Minimal linear-algebra primitives. Keeps the project framework-free and
 * makes the multi-task MLP fully auditable. Performance is acceptable for the
 * model sizes used in this paper (hidden=64-128, batch=256). For larger scale,
 * swap to ND4J or DJL.
 */
final class LinAlg {

    private LinAlg() {}

    /** Out = A * B; A is (m x k), B is (k x n). */
    static double[][] matMul(double[][] A, double[][] B) {
        int m = A.length;
        int k = A[0].length;
        int n = B[0].length;
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++) {
            double[] Ai = A[i];
            double[] Ci = C[i];
            for (int kk = 0; kk < k; kk++) {
                double a = Ai[kk];
                if (a == 0) continue;
                double[] Bk = B[kk];
                for (int j = 0; j < n; j++) {
                    Ci[j] += a * Bk[j];
                }
            }
        }
        return C;
    }

    /** Add a row vector b to each row of M (in-place). */
    static void addRow(double[][] M, double[] b) {
        for (double[] row : M) {
            for (int j = 0; j < b.length; j++) row[j] += b[j];
        }
    }

    /** Element-wise ReLU (in-place). */
    static void relu(double[][] M) {
        for (double[] row : M) {
            for (int j = 0; j < row.length; j++) {
                if (row[j] < 0) row[j] = 0;
            }
        }
    }

    /** Element-wise sigmoid. */
    static double[] sigmoid(double[] x) {
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) y[i] = sigmoid(x[i]);
        return y;
    }

    static double sigmoid(double z) {
        if (z >= 0) {
            double e = Math.exp(-z);
            return 1.0 / (1.0 + e);
        } else {
            double e = Math.exp(z);
            return e / (1.0 + e);
        }
    }

    /** Numerically stable softmax for a single vector. */
    static double[] softmax(double[] x) {
        double max = x[0];
        for (double v : x) if (v > max) max = v;
        double sum = 0;
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = Math.exp(x[i] - max);
            sum += y[i];
        }
        for (int i = 0; i < x.length; i++) y[i] /= sum;
        return y;
    }

    /** Numerically stable softmax for a batch (each row). */
    static double[][] softmax(double[][] X) {
        int n = X.length;
        double[][] Y = new double[n][];
        for (int i = 0; i < n; i++) Y[i] = softmax(X[i]);
        return Y;
    }

    /** Transpose an (m x n) matrix to (n x m). */
    static double[][] transpose(double[][] M) {
        int m = M.length;
        int n = M[0].length;
        double[][] T = new double[n][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) T[j][i] = M[i][j];
        }
        return T;
    }

    /** Sum across rows of M (column-wise reduction) -> length-n vector. */
    static double[] columnSum(double[][] M) {
        int n = M[0].length;
        double[] s = new double[n];
        for (double[] row : M) {
            for (int j = 0; j < n; j++) s[j] += row[j];
        }
        return s;
    }
}
