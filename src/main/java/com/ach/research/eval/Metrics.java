package com.ach.research.eval;

import java.util.Arrays;
import java.util.Comparator;

/**
 * Implementation of standard binary classification metrics. Pure Java; no
 * external dependencies. AUROC and AUPRC use the trapezoidal rule on the
 * sorted score-by-label arrays.
 *
 * Threshold-dependent metrics (precision/recall/F1) are reported at multiple
 * operating points to surface the precision-recall trade-off, which is
 * essential for class-imbalanced fraud and compliance tasks.
 */
public final class Metrics {

    private Metrics() {}

    public record BinaryMetrics(
            double auroc,
            double auprc,
            double bestF1,
            double bestF1Threshold,
            double precisionAtBestF1,
            double recallAtBestF1,
            double prevalence
    ) {}

    public static BinaryMetrics evaluate(double[] scores, int[] labels) {
        if (scores.length != labels.length) {
            throw new IllegalArgumentException("scores and labels must align");
        }
        int n = scores.length;

        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        // Sort ASCENDING by score: lowest score -> rank 1, highest -> rank N.
        // This is required by the Mann-Whitney rank-sum AUROC formula on line 60.
        // (PR-curve and best-F1 sweep below need descending order, so we use a
        // separate descending-order array for that pass.)
        Arrays.sort(order, Comparator.comparingDouble((Integer i) -> scores[i]));

        // AUROC (Mann-Whitney) and AUPRC via numerical integration of PR curve
        int totalPos = 0;
        for (int y : labels) totalPos += y;
        int totalNeg = n - totalPos;
        double prevalence = totalPos / (double) n;

        // AUROC
        double auc = 0;
        if (totalPos > 0 && totalNeg > 0) {
            // Equivalent computation: rank-sum
            double[] ranks = new double[n];
            int idx = 0;
            while (idx < n) {
                int j = idx;
                while (j < n - 1 && scores[order[j + 1]] == scores[order[idx]]) j++;
                double avgRank = (idx + j + 2) / 2.0;
                for (int k = idx; k <= j; k++) ranks[order[k]] = avgRank;
                idx = j + 1;
            }
            double rankSumPos = 0;
            for (int i = 0; i < n; i++) if (labels[i] == 1) rankSumPos += ranks[i];
            auc = (rankSumPos - totalPos * (totalPos + 1) / 2.0) / ((double) totalPos * totalNeg);
        }

        // AUPRC, best F1 sweep — needs DESCENDING order (highest score first).
        Integer[] descOrder = new Integer[n];
        for (int i = 0; i < n; i++) descOrder[i] = i;
        Arrays.sort(descOrder, Comparator.comparingDouble((Integer i) -> -scores[i]));

        int tp = 0, fp = 0;
        double prevRecall = 0, prevPrec = 1;
        double auprc = 0;
        double bestF1 = 0, bestThr = 0.5, bestPrec = 0, bestRec = 0;

        for (int k = 0; k < n; k++) {
            int i = descOrder[k];
            if (labels[i] == 1) tp++; else fp++;
            double precision = tp / (double) (tp + fp);
            double recall = totalPos > 0 ? tp / (double) totalPos : 0;
            double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0;
            if (f1 > bestF1) {
                bestF1 = f1;
                bestThr = scores[i];
                bestPrec = precision;
                bestRec = recall;
            }
            // Trapezoidal AUPRC step
            auprc += (recall - prevRecall) * (precision + prevPrec) / 2;
            prevRecall = recall;
            prevPrec = precision;
        }

        return new BinaryMetrics(auc, auprc, bestF1, bestThr, bestPrec, bestRec, prevalence);
    }
}
