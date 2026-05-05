package com.ach.research.data;

import java.util.List;

/**
 * Feature engineering. Produces an 18-dimensional vector per transaction.
 *
 * IMPORTANT: feature ordering and definitions are part of the model contract.
 * Any change here MUST be replicated in the inference path
 * ({@link com.ach.research.service.DecisioningService}) to avoid skew.
 */
public final class FeatureEngineer {

    private FeatureEngineer() {}

    /**
     * Vectorize a single transaction.
     */
    public static FeatureVector engineer(Transaction t) {
        double[] f = new double[FeatureVector.DIM];

        f[0] = t.amount();
        f[1] = Math.log1p(t.amount());
        f[2] = t.origBalanceBefore();
        f[3] = t.origBalanceAfter();
        f[4] = t.destBalanceBefore();
        f[5] = t.destBalanceAfter();

        double balanceDeltaOrig = t.origBalanceAfter() - t.origBalanceBefore();
        double balanceDeltaDest = t.destBalanceAfter() - t.destBalanceBefore();
        f[6] = balanceDeltaOrig;
        f[7] = balanceDeltaDest;
        f[8] = Math.abs(balanceDeltaOrig + balanceDeltaDest);

        f[9]  = t.amount() / (t.origBalanceBefore() + 1.0);
        f[10] = t.amount() / (t.destBalanceBefore() + 1.0);

        f[11] = t.step() % 24;
        int dayOfWeek = (t.step() / 24) % 7;
        f[12] = (dayOfWeek == 5 || dayOfWeek == 6) ? 1.0 : 0.0;

        // One-hot for paymentType: ordinal positions 13..17
        int ord = t.paymentType().ordinal();
        f[13 + ord] = 1.0;

        return new FeatureVector(f);
    }

    /**
     * Engineer a batch and produce a {@link LabeledDataset}.
     */
    public static LabeledDataset engineerBatch(List<Transaction> txs) {
        int n = txs.size();
        double[][] features = new double[n][];
        int[] fraud = new int[n];
        int[] returnCode = new int[n];
        int[] anyReturn = new int[n];
        int[] compliance = new int[n];
        long[] tids = new long[n];

        for (int i = 0; i < n; i++) {
            Transaction t = txs.get(i);
            features[i] = engineer(t).features();
            fraud[i] = t.isFraud();
            returnCode[i] = t.returnCode().ordinal();
            anyReturn[i] = t.anyReturnLabel();
            compliance[i] = t.complianceFlag();
            tids[i] = t.transactionId();
        }
        return new LabeledDataset(features, fraud, returnCode, anyReturn, compliance, tids);
    }
}
