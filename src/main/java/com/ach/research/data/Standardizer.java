package com.ach.research.data;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;

/**
 * Z-score standardization fit on the training partition. The fitted mean/std
 * arrays are persisted (JSON) and consumed at inference time to maintain
 * train-serve consistency.
 *
 * Fitting is intentionally on training data only to prevent test-set leakage.
 */
public final class Standardizer {

    public double[] mean;
    public double[] std;

    public Standardizer() {}

    public Standardizer(double[] mean, double[] std) {
        this.mean = mean;
        this.std = std;
    }

    public void fit(double[][] data) {
        int dim = data[0].length;
        mean = new double[dim];
        std = new double[dim];

        for (double[] row : data) {
            for (int j = 0; j < dim; j++) mean[j] += row[j];
        }
        for (int j = 0; j < dim; j++) mean[j] /= data.length;

        for (double[] row : data) {
            for (int j = 0; j < dim; j++) {
                double d = row[j] - mean[j];
                std[j] += d * d;
            }
        }
        for (int j = 0; j < dim; j++) {
            std[j] = Math.sqrt(std[j] / data.length);
            if (std[j] < 1e-8) std[j] = 1.0;       // avoid division by zero
        }
    }

    public double[][] transform(double[][] data) {
        int n = data.length;
        int dim = data[0].length;
        double[][] out = new double[n][dim];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < dim; j++) {
                out[i][j] = (data[i][j] - mean[j]) / std[j];
            }
        }
        return out;
    }

    public double[] transform(double[] x) {
        double[] out = new double[x.length];
        for (int j = 0; j < x.length; j++) out[j] = (x[j] - mean[j]) / std[j];
        return out;
    }

    public void save(File f) throws IOException {
        new ObjectMapper().writerWithDefaultPrettyPrinter().writeValue(f, this);
    }

    public static Standardizer load(File f) throws IOException {
        return new ObjectMapper().readValue(f, Standardizer.class);
    }
}
