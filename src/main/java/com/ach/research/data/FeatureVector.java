package com.ach.research.data;

/**
 * Engineered feature vector used as model input. 18 dimensions total:
 *
 *   0: amount (raw)
 *   1: log(amount + 1)
 *   2: origBalanceBefore
 *   3: origBalanceAfter
 *   4: destBalanceBefore
 *   5: destBalanceAfter
 *   6: balanceDeltaOrig (origBalanceAfter - origBalanceBefore)
 *   7: balanceDeltaDest (destBalanceAfter - destBalanceBefore)
 *   8: balanceConsistency (|balanceDeltaOrig + balanceDeltaDest|)
 *   9: amountToBalanceRatioOrig (amount / (origBalanceBefore + 1))
 *  10: amountToBalanceRatioDest (amount / (destBalanceBefore + 1))
 *  11: stepOfDay (step % 24)
 *  12: isWeekend (1 if step / 24 % 7 in [5,6])
 *  13: paymentType_DEBIT one-hot
 *  14: paymentType_CREDIT one-hot
 *  15: paymentType_CASH_OUT one-hot
 *  16: paymentType_TRANSFER one-hot
 *  17: paymentType_PAYMENT one-hot
 *
 * The Java FeatureEngineer must stay in lockstep with the Python equivalent.
 */
public record FeatureVector(double[] features) {

    public static final int DIM = 18;

    public FeatureVector {
        if (features.length != DIM) {
            throw new IllegalArgumentException(
                "FeatureVector dimension must be " + DIM + " but got " + features.length);
        }
    }

    public double get(int i) {
        return features[i];
    }
}
