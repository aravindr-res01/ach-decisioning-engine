package com.ach.research.data;

/**
 * Materialized dataset: features (N x D) plus three task label vectors. Used
 * by trainers and evaluators. All arrays are aligned by row index.
 */
public record LabeledDataset(
        double[][] features,         // [N][DIM]
        int[] fraudLabels,           // [N], 0/1
        int[] returnCodeLabels,      // [N], 0..11 (NACHA index)
        int[] anyReturnLabels,       // [N], 0/1 (derived: 1 if returnCodeLabels[i] != 0)
        int[] complianceLabels,      // [N], 0/1
        long[] transactionIds        // [N]
) {

    public int size() {
        return features.length;
    }

    public int featureDim() {
        return features[0].length;
    }

    /** Slice rows [from, to). */
    public LabeledDataset slice(int from, int to) {
        int n = to - from;
        double[][] f = new double[n][];
        int[] fr = new int[n], rc = new int[n], ar = new int[n], cp = new int[n];
        long[] tid = new long[n];
        for (int i = 0; i < n; i++) {
            f[i] = features[from + i];
            fr[i] = fraudLabels[from + i];
            rc[i] = returnCodeLabels[from + i];
            ar[i] = anyReturnLabels[from + i];
            cp[i] = complianceLabels[from + i];
            tid[i] = transactionIds[from + i];
        }
        return new LabeledDataset(f, fr, rc, ar, cp, tid);
    }
}
