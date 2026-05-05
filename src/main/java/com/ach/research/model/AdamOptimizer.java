package com.ach.research.model;

/**
 * Adam optimizer (Kingma & Ba, 2015). Maintains first/second moment buffers
 * for each parameter tensor it owns. Created per parameter; centralized
 * stepping happens in {@link MultiTaskMlp#step}.
 */
final class AdamOptimizer {

    private final double lr;
    private final double beta1;
    private final double beta2;
    private final double eps;
    private int t;

    private double[][] m2;     // 2D moment-1
    private double[][] v2;
    private double[]   m1;     // 1D moment-1
    private double[]   v1;

    AdamOptimizer(double lr, double beta1, double beta2, double eps,
                  int rows, int cols) {
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.t = 0;
        if (cols > 0) {
            this.m2 = new double[rows][cols];
            this.v2 = new double[rows][cols];
        } else {
            this.m1 = new double[rows];
            this.v1 = new double[rows];
        }
    }

    /** Update a 2D parameter in-place given its gradient (same shape). */
    void update(double[][] param, double[][] grad) {
        t++;
        double bc1 = 1.0 - Math.pow(beta1, t);
        double bc2 = 1.0 - Math.pow(beta2, t);
        for (int i = 0; i < param.length; i++) {
            for (int j = 0; j < param[0].length; j++) {
                m2[i][j] = beta1 * m2[i][j] + (1 - beta1) * grad[i][j];
                v2[i][j] = beta2 * v2[i][j] + (1 - beta2) * grad[i][j] * grad[i][j];
                double mh = m2[i][j] / bc1;
                double vh = v2[i][j] / bc2;
                param[i][j] -= lr * mh / (Math.sqrt(vh) + eps);
            }
        }
    }

    /** Update a 1D bias vector in-place. */
    void update(double[] param, double[] grad) {
        t++;
        double bc1 = 1.0 - Math.pow(beta1, t);
        double bc2 = 1.0 - Math.pow(beta2, t);
        for (int i = 0; i < param.length; i++) {
            m1[i] = beta1 * m1[i] + (1 - beta1) * grad[i];
            v1[i] = beta2 * v1[i] + (1 - beta2) * grad[i] * grad[i];
            double mh = m1[i] / bc1;
            double vh = v1[i] / bc2;
            param[i] -= lr * mh / (Math.sqrt(vh) + eps);
        }
    }
}
