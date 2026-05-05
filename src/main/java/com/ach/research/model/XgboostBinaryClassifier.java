package com.ach.research.model;

import com.ach.research.data.LabeledDataset;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * XGBoost binary classifier wrapper, used as the strongest baseline for each
 * task (fraud, anyReturn, compliance). Three independent boosters are trained;
 * inference latency is therefore ~3x a single-model multi-task forward pass,
 * which is one of the empirical points the paper makes.
 *
 * Hyperparameters chosen via small grid search on validation set:
 *   eta=0.1, max_depth=6, min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
 *   eval_metric=auc, objective=binary:logistic.
 */
public final class XgboostBinaryClassifier {

    private Booster booster;

    public void fit(double[][] X, int[] y, int numRound, long seed) throws XGBoostError {
        DMatrix dtrain = toDMatrix(X, y);
        Map<String, Object> params = baseParams(seed);
        booster = XGBoost.train(dtrain, params, numRound, new HashMap<>(), null, null);
    }

    public double[] predictProba(double[][] X) throws XGBoostError {
        DMatrix d = toDMatrix(X, null);
        float[][] raw = booster.predict(d);
        double[] out = new double[raw.length];
        for (int i = 0; i < raw.length; i++) out[i] = raw[i][0];
        return out;
    }

    public double predictProba(double[] x) throws XGBoostError {
        return predictProba(new double[][]{x})[0];
    }

    public void save(String path) throws XGBoostError, IOException {
        booster.saveModel(path);
    }

    public void load(String path) throws XGBoostError {
        booster = XGBoost.loadModel(path);
    }

    private static Map<String, Object> baseParams(long seed) {
        Map<String, Object> p = new HashMap<>();
        p.put("eta", 0.1);
        p.put("max_depth", 6);
        p.put("min_child_weight", 1);
        p.put("subsample", 0.8);
        p.put("colsample_bytree", 0.8);
        p.put("objective", "binary:logistic");
        p.put("eval_metric", "auc");
        p.put("seed", seed);
        p.put("verbosity", 0);
        p.put("nthread", 4);
        return p;
    }

    private static DMatrix toDMatrix(double[][] X, int[] y) throws XGBoostError {
        int n = X.length;
        int dim = X[0].length;
        float[] flat = new float[n * dim];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < dim; j++) {
                flat[i * dim + j] = (float) X[i][j];
            }
        }
        DMatrix d = new DMatrix(flat, n, dim, Float.NaN);
        if (y != null) {
            float[] labels = new float[y.length];
            for (int i = 0; i < y.length; i++) labels[i] = y[i];
            d.setLabel(labels);
        }
        return d;
    }

    /**
     * Convenience trainer for the three-task baseline: returns three fitted
     * boosters (fraud, anyReturn, compliance). Used by ExperimentRunner.
     */
    public static XgboostBinaryClassifier[] fitThreeTask(LabeledDataset train, int numRound, long seed)
            throws XGBoostError {
        XgboostBinaryClassifier fraud = new XgboostBinaryClassifier();
        XgboostBinaryClassifier anyReturn = new XgboostBinaryClassifier();
        XgboostBinaryClassifier compliance = new XgboostBinaryClassifier();
        fraud.fit(train.features(), train.fraudLabels(), numRound, seed);
        anyReturn.fit(train.features(), train.anyReturnLabels(), numRound, seed);
        compliance.fit(train.features(), train.complianceLabels(), numRound, seed);
        return new XgboostBinaryClassifier[]{fraud, anyReturn, compliance};
    }
}
