package com.ach.research.eval;

/**
 * Per-task, per-model threshold calibration on the validation set.
 *
 * Each model's task-head outputs are scored at a grid of thresholds
 * tau in [0.001, 0.999]; the threshold minimizing per-task Operational
 * Cost on the validation split is selected and then applied to the
 * test split for the headline cost computation. This is standard
 * practice for asymmetric-cost classification and is required for a
 * fair comparison between models whose probability outputs have
 * different calibration characteristics (e.g., gradient-boosted trees
 * tend to produce sharply bimodal outputs while sigmoid-headed neural
 * networks produce more diffuse distributions).
 *
 * Without this calibration step, comparing models at a single fixed
 * threshold (e.g., 0.5) conflates predictive ordering with output
 * calibration, biasing the cost comparison toward whichever model
 * happens to have probability outputs concentrated near the chosen
 * threshold.
 */
public final class ThresholdCalibrator {

    /** A single calibrated operating point for one task. */
    public record CalibratedThreshold(double threshold, double validationCost) {}

    private static final int N_GRID = 199;          // thresholds 0.005 .. 0.995 step 0.005
    private static final double GRID_LO = 0.005;
    private static final double GRID_HI = 0.995;

    private ThresholdCalibrator() {}

    /**
     * Sweep thresholds on the validation set and return the one that
     * minimizes Operational Cost for a given (probability, label) task pair.
     */
    public static CalibratedThreshold calibrate(
            double[] valProbs, int[] valLabels, OperationalCost.CostMatrix costs) {

        double bestThr = 0.5;
        double bestCost = Double.POSITIVE_INFINITY;

        for (int g = 0; g < N_GRID; g++) {
            double thr = GRID_LO + (GRID_HI - GRID_LO) * g / (N_GRID - 1);
            double cost = costAtThreshold(valProbs, valLabels, thr, costs);
            if (cost < bestCost) {
                bestCost = cost;
                bestThr = thr;
            }
        }
        return new CalibratedThreshold(bestThr, bestCost);
    }

    /** Compute Operational Cost at a given threshold for a single task. */
    public static double costAtThreshold(
            double[] probs, int[] labels, double thr, OperationalCost.CostMatrix cm) {
        double cost = 0;
        for (int i = 0; i < probs.length; i++) {
            int pred = probs[i] >= thr ? 1 : 0;
            if (pred == 0 && labels[i] == 1) cost += cm.fnCost();
            else if (pred == 1 && labels[i] == 0) cost += cm.fpCost();
        }
        return cost;
    }
}