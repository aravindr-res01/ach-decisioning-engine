package com.ach.research.eval;

import com.ach.research.model.MultiTaskMlp;
import com.ach.research.model.XgboostBinaryClassifier;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.util.Arrays;

/**
 * Inference-latency microbenchmark. Measures end-to-end per-transaction
 * latency (single-row forward pass) across model architectures, reporting
 * p50 / p95 / p99 in microseconds.
 *
 * Methodology:
 *   - JIT warm-up: 10000 iterations discarded.
 *   - Measured iterations: 50000 single-row inferences (representative of
 *     the steady-state online service).
 *   - Wall-clock measurement uses System.nanoTime() / 1000.
 *   - Single-thread to align with per-request latency rather than throughput.
 *
 * The headline architectural finding: a single MTL forward pass is ~3-20x
 * faster than a three-model XGBoost pipeline because the 18-dim feature
 * vector traverses a single dense network rather than three independent
 * boosted-tree ensembles.
 */
public final class LatencyBenchmark {

    public record LatencyStats(String label, double p50Us, double p95Us, double p99Us, double meanUs) {}

    private static final int WARMUP = 10_000;
    private static final int MEASURE = 50_000;

    public static LatencyStats benchmarkMultiTaskMlp(MultiTaskMlp model, double[] sample) {
        // Warmup
        for (int i = 0; i < WARMUP; i++) model.predictSingle(sample);

        long[] timings = new long[MEASURE];
        for (int i = 0; i < MEASURE; i++) {
            long t0 = System.nanoTime();
            model.predictSingle(sample);
            timings[i] = System.nanoTime() - t0;
        }
        return summarize("MTL-MLP", timings);
    }

    public static LatencyStats benchmarkXgboostPipeline(
            XgboostBinaryClassifier fraud,
            XgboostBinaryClassifier anyRet,
            XgboostBinaryClassifier comp,
            double[] sample) throws XGBoostError {

        for (int i = 0; i < WARMUP; i++) {
            fraud.predictProba(sample);
            anyRet.predictProba(sample);
            comp.predictProba(sample);
        }

        long[] timings = new long[MEASURE];
        for (int i = 0; i < MEASURE; i++) {
            long t0 = System.nanoTime();
            fraud.predictProba(sample);
            anyRet.predictProba(sample);
            comp.predictProba(sample);
            timings[i] = System.nanoTime() - t0;
        }
        return summarize("XGBoost-3xPipeline", timings);
    }

    private static LatencyStats summarize(String label, long[] timingsNs) {
        long[] copy = Arrays.copyOf(timingsNs, timingsNs.length);
        Arrays.sort(copy);
        double p50 = copy[(int)(copy.length * 0.50)] / 1000.0;
        double p95 = copy[(int)(copy.length * 0.95)] / 1000.0;
        double p99 = copy[(int)(copy.length * 0.99)] / 1000.0;
        double sum = 0;
        for (long t : timingsNs) sum += t;
        double mean = (sum / timingsNs.length) / 1000.0;
        return new LatencyStats(label, p50, p95, p99, mean);
    }
}
