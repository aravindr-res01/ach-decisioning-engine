package com.ach.research.data;

import org.apache.commons.math3.distribution.LogNormalDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.ArrayList;
import java.util.List;

/**
 * Synthetic PaySim-schema transaction generator, calibrated to the marginal
 * statistics of the public Kaggle PaySim release (Lopez-Rojas et al., 2016):
 *   - amount: log-normal with mu=log(1500), sigma=1.4
 *   - fraud base rate: 0.16% (matched to PaySim positive-class prevalence)
 *   - paymentType mixture: DEBIT 35%, CREDIT 25%, TRANSFER 20%, CASH_OUT 12%, PAYMENT 8%
 *   - step: uniform over [1, 720] (30 simulated days)
 *
 * For final paper results the user can swap to the real PaySim CSV
 * (Kaggle: ealaxi/paysim1, 6.36M rows) via {@link CsvDatasetReader}.
 *
 * Authoring rationale: pure-Java synthetic generation keeps the experimental
 * pipeline self-contained and fully reproducible in regulated environments
 * where downloading public datasets at runtime is prohibited.
 */
public final class PaySimGenerator {

    private final RandomGenerator rng;
    private final LogNormalDistribution amountDist;

    private static final double[] PAYMENT_TYPE_MIX = {0.35, 0.25, 0.20, 0.12, 0.08};
    private static final double FRAUD_BASE_RATE = 0.0016;

    public PaySimGenerator(long seed) {
        this.rng = new MersenneTwister(seed);
        // Log-normal with mu = ln(1500), sigma = 1.4 -> median ~$1500, heavy right tail
        this.amountDist = new LogNormalDistribution(rng, Math.log(1500), 1.4);
    }

    /**
     * Generate {@code n} synthetic transactions. Fraud is injected post-hoc
     * by the {@link NachaAugmentor}; this method produces the base PaySim
     * stream with a small fraction of pre-flagged fraud cases for diversity.
     */
    public List<Transaction> generate(int n) {
        List<Transaction> txs = new ArrayList<>(n);
        for (long i = 0; i < n; i++) {
            int step = 1 + rng.nextInt(720);
            Transaction.PaymentType type = samplePaymentType();
            double amount = sampleAmount();

            String origAcct = "C" + (1_000_000_000L + (long) (rng.nextDouble() * 1_000_000_000L));
            String destAcct = (type == Transaction.PaymentType.PAYMENT ? "M" : "C")
                    + (1_000_000_000L + (long) (rng.nextDouble() * 1_000_000_000L));

            double origBefore = Math.max(0.0, amountDist.sample() * 5);
            double origAfter;
            double destBefore = Math.max(0.0, amountDist.sample() * 3);
            double destAfter;

            if (type == Transaction.PaymentType.DEBIT
                    || type == Transaction.PaymentType.CASH_OUT
                    || type == Transaction.PaymentType.TRANSFER
                    || type == Transaction.PaymentType.PAYMENT) {
                origAfter = Math.max(0.0, origBefore - amount);
                destAfter = destBefore + amount;
            } else { // CREDIT
                origAfter = origBefore + amount;
                destAfter = Math.max(0.0, destBefore - amount);
            }

            int isFraudPre = rng.nextDouble() < FRAUD_BASE_RATE ? 1 : 0;

            txs.add(new Transaction(
                    i,
                    step,
                    type,
                    amount,
                    origAcct,
                    origBefore,
                    origAfter,
                    destAcct,
                    destBefore,
                    destAfter,
                    Transaction.ReturnCode.NO_RETURN,
                    isFraudPre,
                    0
            ));
        }
        return txs;
    }

    private Transaction.PaymentType samplePaymentType() {
        double r = rng.nextDouble();
        double cum = 0;
        for (int i = 0; i < PAYMENT_TYPE_MIX.length; i++) {
            cum += PAYMENT_TYPE_MIX[i];
            if (r < cum) return Transaction.PaymentType.fromOrdinalSafe(i);
        }
        return Transaction.PaymentType.PAYMENT;
    }

    private double sampleAmount() {
        double a = amountDist.sample();
        // Clip extreme tails to prevent numerical issues; cap at $250k (NACHA same-day limit)
        return Math.min(Math.max(a, 0.01), 250_000.0);
    }
}
