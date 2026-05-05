package com.ach.research.runner;

import com.ach.research.data.*;
import com.ach.research.eval.LatencyBenchmark;
import com.ach.research.eval.Metrics;
import com.ach.research.eval.OperationalCost;
import com.ach.research.eval.ThresholdCalibrator;
import com.ach.research.model.MultiTaskMlp;
import com.ach.research.model.XgboostBinaryClassifier;
import com.ach.research.viz.FigureFactory;
import com.fasterxml.jackson.databind.ObjectMapper;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Reproduces the full experimental evaluation reported in the manuscript.
 *
 * Workflow:
 *   1. Generate synthetic PaySim-schema transactions (configurable size).
 *   2. NACHA-augment (return codes + BSA/AML compliance flags).
 *   3. Engineer features and standardize.
 *   4. Temporal train/val/test split (70/15/15) by step.
 *   5. Train 3 model families across 2 seeds:
 *        - XGBoost (3 separate boosters per task, the strongest baseline)
 *        - MTL-MLP-UNIFORM (shared encoder, equal-weight loss)
 *        - MTL-MLP-KENDALL (shared encoder, learned uncertainty weighting)
 *   6. Compute AUROC, AUPRC, F1, Operational Cost on test set.
 *   7. Benchmark single-transaction inference latency (p50/p95/p99).
 *   8. Render seven publication figures.
 *   9. Persist results JSON for paper inclusion and reproducibility.
 *
 * Run with:  java -jar ach-decisioning.jar --mode=experiment
 */
public final class ExperimentRunner {

    private static final int NUM_RETURN_CLASSES = Transaction.ReturnCode.values().length;
    private static final long[] SEEDS = {42L, 2025L};
    private static final int XGBOOST_ROUNDS = 100;

    /** Run-time configuration parsed from command-line flags. */
    public record RunConfig(DataSource source, int sampleSize, java.nio.file.Path paysimPath) {
        public static RunConfig defaults() {
            return new RunConfig(DataSource.SYNTHETIC, 100_000, null);
        }
    }

    public enum DataSource { SYNTHETIC, PAYSIM }

    private ExperimentRunner() {}

    /** Backward-compatible entry point used by main(). */
    public static void runFullPipeline() {
        runFullPipeline(RunConfig.defaults());
    }

    /** Parse command-line flags into a RunConfig. */
    public static RunConfig parseArgs(String[] args) {
        DataSource src = DataSource.SYNTHETIC;
        int n = 100_000;
        java.nio.file.Path path = null;
        for (String a : args) {
            if (a.startsWith("--data=")) {
                src = DataSource.valueOf(a.substring("--data=".length()).toUpperCase());
            } else if (a.startsWith("--data-path=")) {
                path = java.nio.file.Paths.get(a.substring("--data-path=".length()));
            } else if (a.startsWith("--sample-size=")) {
                String v = a.substring("--sample-size=".length());
                n = "ALL".equalsIgnoreCase(v) ? Integer.MAX_VALUE : Integer.parseInt(v);
            }
        }
        if (src == DataSource.PAYSIM && path == null) {
            throw new IllegalArgumentException(
                    "--data=paysim requires --data-path=/path/to/PS_*_log.csv " +
                    "(download via: kaggle datasets download -d ealaxi/paysim1 -p ./data --unzip)");
        }
        return new RunConfig(src, n, path);
    }

    public static void runFullPipeline(RunConfig cfg) {
        try {
            System.out.println("[Pipeline] Java/Spring Boot ACH Decisioning - Experiment Run");
            System.out.println("===============================================================");
            File outDir = new File("output");
            outDir.mkdirs();
            File figDir = new File(outDir, "figures");
            figDir.mkdirs();

            // --- 1-4. Dataset construction ---
            long t0 = System.currentTimeMillis();
            List<Transaction> raw;
            if (cfg.source() == DataSource.PAYSIM) {
                System.out.printf("[1/9] Loading Kaggle PaySim from %s (sample-size=%d)...%n",
                        cfg.paysimPath(), cfg.sampleSize());
                raw = PaysimCsvReader.read(cfg.paysimPath(), cfg.sampleSize(), SEEDS[0]);
                System.out.printf("    Loaded %d real PaySim transactions.%n", raw.size());
            } else {
                System.out.printf("[1/9] Generating %d synthetic PaySim-schema transactions...%n",
                        cfg.sampleSize());
                PaySimGenerator gen = new PaySimGenerator(SEEDS[0]);
                raw = gen.generate(cfg.sampleSize());
            }

            System.out.println("[2/9] NACHA augmentation (return codes + compliance)...");
            NachaAugmentor aug = new NachaAugmentor(SEEDS[0]);
            List<Transaction> txs = aug.augment(raw);

            System.out.println("[3/9] Feature engineering...");
            LabeledDataset full = FeatureEngineer.engineerBatch(txs);

            System.out.println("[4/9] Temporal split (70/15/15)...");
            int n = full.size();
            int trainEnd = (int) (n * 0.70);
            int valEnd   = (int) (n * 0.85);
            // Sort by step is implicit because PaySimGenerator emits in step order
            LabeledDataset trainRaw = full.slice(0, trainEnd);
            LabeledDataset valRaw   = full.slice(trainEnd, valEnd);
            LabeledDataset testRaw  = full.slice(valEnd, n);

            // Standardize on training only, then transform
            Standardizer scaler = new Standardizer();
            scaler.fit(trainRaw.features());
            LabeledDataset train = standardize(trainRaw, scaler);
            LabeledDataset val   = standardize(valRaw,   scaler);
            LabeledDataset test  = standardize(testRaw,  scaler);
            scaler.save(new File(outDir, "scaler.json"));

            System.out.printf("    Sizes: train=%d val=%d test=%d  fraud-prevalence(test)=%.4f%n",
                    train.size(), val.size(), test.size(),
                    prevalence(test.fraudLabels()));

            // --- 5-6. Train models across seeds ---
            System.out.println("[5/9] Training models across seeds...");

            Map<String, List<TaskMetrics>> byModel = new LinkedHashMap<>();
            Map<String, List<OperationalCost.CostBreakdown>> byModelCost = new LinkedHashMap<>();
            Map<String, Double> totalTrainTimeByModel = new LinkedHashMap<>();
            Map<String, Map<String, ThresholdCalibrator.CalibratedThreshold>> byModelThresholds = new LinkedHashMap<>();

            for (long seed : SEEDS) {
                System.out.printf("  Seed %d:%n", seed);

                // ---- XGBoost baseline (3 boosters) ----
                long ts = System.currentTimeMillis();
                XgboostBinaryClassifier[] xgb = XgboostBinaryClassifier.fitThreeTask(train, XGBOOST_ROUNDS, seed);
                long xgbTime = System.currentTimeMillis() - ts;

                // Test-set predictions for AUROC/AUPRC
                double[] xgbTestF = xgb[0].predictProba(test.features());
                double[] xgbTestA = xgb[1].predictProba(test.features());
                double[] xgbTestC = xgb[2].predictProba(test.features());
                // Validation-set predictions for threshold calibration
                double[] xgbValF = xgb[0].predictProba(val.features());
                double[] xgbValA = xgb[1].predictProba(val.features());
                double[] xgbValC = xgb[2].predictProba(val.features());

                record(byModel, "XGBoost", new TaskMetrics(
                        Metrics.evaluate(xgbTestF, test.fraudLabels()),
                        Metrics.evaluate(xgbTestA, test.anyReturnLabels()),
                        Metrics.evaluate(xgbTestC, test.complianceLabels())
                ));
                Map<String, ThresholdCalibrator.CalibratedThreshold> xgbThr = new LinkedHashMap<>();
                record(byModelCost, "XGBoost", costWithCalibration(
                        xgbValF, xgbValA, xgbValC,
                        val.fraudLabels(), val.anyReturnLabels(), val.complianceLabels(),
                        xgbTestF, xgbTestA, xgbTestC,
                        test.fraudLabels(), test.anyReturnLabels(), test.complianceLabels(),
                        xgbThr));
                byModelThresholds.put("XGBoost", xgbThr);
                addTime(totalTrainTimeByModel, "XGBoost", xgbTime);

                // ---- MTL-Uniform ----
                ts = System.currentTimeMillis();
                MultiTaskMlp.Config cfgU = MultiTaskMlp.Config.defaults(
                        train.featureDim(), NUM_RETURN_CLASSES, MultiTaskMlp.LossWeighting.UNIFORM, seed);
                MultiTaskMlp mtlU = new MultiTaskMlp(cfgU);
                mtlU.fit(train);
                long mtlUTime = System.currentTimeMillis() - ts;

                MultiTaskMlp.Predictions pUtest = mtlU.predict(test.features());
                MultiTaskMlp.Predictions pUval  = mtlU.predict(val.features());
                record(byModel, "MTL-Uniform", new TaskMetrics(
                        Metrics.evaluate(pUtest.fraud(),     test.fraudLabels()),
                        Metrics.evaluate(pUtest.anyReturn(), test.anyReturnLabels()),
                        Metrics.evaluate(pUtest.compliance(),test.complianceLabels())
                ));
                Map<String, ThresholdCalibrator.CalibratedThreshold> mtlUThr = new LinkedHashMap<>();
                record(byModelCost, "MTL-Uniform", costWithCalibration(
                        pUval.fraud(), pUval.anyReturn(), pUval.compliance(),
                        val.fraudLabels(), val.anyReturnLabels(), val.complianceLabels(),
                        pUtest.fraud(), pUtest.anyReturn(), pUtest.compliance(),
                        test.fraudLabels(), test.anyReturnLabels(), test.complianceLabels(),
                        mtlUThr));
                byModelThresholds.put("MTL-Uniform", mtlUThr);
                addTime(totalTrainTimeByModel, "MTL-Uniform", mtlUTime);

                // ---- MTL-Kendall ----
                ts = System.currentTimeMillis();
                MultiTaskMlp.Config cfgK = MultiTaskMlp.Config.defaults(
                        train.featureDim(), NUM_RETURN_CLASSES, MultiTaskMlp.LossWeighting.KENDALL, seed);
                MultiTaskMlp mtlK = new MultiTaskMlp(cfgK);
                mtlK.fit(train);
                long mtlKTime = System.currentTimeMillis() - ts;

                MultiTaskMlp.Predictions pKtest = mtlK.predict(test.features());
                MultiTaskMlp.Predictions pKval  = mtlK.predict(val.features());
                record(byModel, "MTL-Kendall", new TaskMetrics(
                        Metrics.evaluate(pKtest.fraud(),     test.fraudLabels()),
                        Metrics.evaluate(pKtest.anyReturn(), test.anyReturnLabels()),
                        Metrics.evaluate(pKtest.compliance(),test.complianceLabels())
                ));
                Map<String, ThresholdCalibrator.CalibratedThreshold> mtlKThr = new LinkedHashMap<>();
                record(byModelCost, "MTL-Kendall", costWithCalibration(
                        pKval.fraud(), pKval.anyReturn(), pKval.compliance(),
                        val.fraudLabels(), val.anyReturnLabels(), val.complianceLabels(),
                        pKtest.fraud(), pKtest.anyReturn(), pKtest.compliance(),
                        test.fraudLabels(), test.anyReturnLabels(), test.complianceLabels(),
                        mtlKThr));
                byModelThresholds.put("MTL-Kendall", mtlKThr);
                addTime(totalTrainTimeByModel, "MTL-Kendall", mtlKTime);

                if (seed == SEEDS[SEEDS.length - 1]) {
                    // Use the last-trained seed for latency benchmarks (held in scope)
                    System.out.println("[7/9] Latency microbenchmark (single-row inference)...");
                    double[] sample = test.features()[0];
                    LatencyBenchmark.LatencyStats lsXgb =
                            LatencyBenchmark.benchmarkXgboostPipeline(xgb[0], xgb[1], xgb[2], sample);
                    LatencyBenchmark.LatencyStats lsK =
                            LatencyBenchmark.benchmarkMultiTaskMlp(mtlK, sample);

                    Map<String, double[]> latencyMap = new LinkedHashMap<>();
                    latencyMap.put("XGBoost-3xPipeline", new double[]{lsXgb.p50Us(), lsXgb.p95Us(), lsXgb.p99Us()});
                    latencyMap.put("MTL-MLP-Kendall",    new double[]{lsK.p50Us(),   lsK.p95Us(),   lsK.p99Us()});
                    FigureFactory.figLatency(latencyMap, new File(figDir, "fig5_latency.png"));

                    System.out.printf("    XGBoost p99=%.0fus | MTL p99=%.0fus | speedup=%.1fx%n",
                            lsXgb.p99Us(), lsK.p99Us(), lsXgb.p99Us() / Math.max(1, lsK.p99Us()));

                    Map<String, Object> latencyJson = new LinkedHashMap<>();
                    latencyJson.put("XGBoost", lsXgb);
                    latencyJson.put("MTL-MLP-Kendall", lsK);
                    new ObjectMapper().writerWithDefaultPrettyPrinter()
                            .writeValue(new File(outDir, "latency.json"), latencyJson);
                }
            }

            // --- 8. Aggregate across seeds (mean) ---
            System.out.println("[8/9] Aggregating metrics across seeds...");
            Map<String, Map<String, Double>> meanAuroc = new LinkedHashMap<>();
            Map<String, double[]> meanCost = new LinkedHashMap<>();
            for (var e : byModel.entrySet()) {
                Map<String, Double> taskMap = new LinkedHashMap<>();
                taskMap.put("Fraud",      mean(e.getValue(), m -> m.fraud.auroc()));
                taskMap.put("AnyReturn",  mean(e.getValue(), m -> m.anyReturn.auroc()));
                taskMap.put("Compliance", mean(e.getValue(), m -> m.compliance.auroc()));
                meanAuroc.put(e.getKey(), taskMap);
            }
            for (var e : byModelCost.entrySet()) {
                double f = e.getValue().stream().mapToDouble(c -> c.fraudCost()).average().orElse(0);
                double r = e.getValue().stream().mapToDouble(c -> c.anyReturnCost()).average().orElse(0);
                double c = e.getValue().stream().mapToDouble(x -> x.complianceCost()).average().orElse(0);
                meanCost.put(e.getKey(), new double[]{f, r, c});
            }

            // --- 9. Render figures + persist results JSON ---
            System.out.println("[9/9] Rendering figures + writing results JSON...");

            Map<String, double[]> taskRates = new LinkedHashMap<>();
            taskRates.put("Fraud",      new double[]{prevalence(train.fraudLabels()), prevalence(val.fraudLabels()), prevalence(test.fraudLabels())});
            taskRates.put("AnyReturn",  new double[]{prevalence(train.anyReturnLabels()), prevalence(val.anyReturnLabels()), prevalence(test.anyReturnLabels())});
            taskRates.put("Compliance", new double[]{prevalence(train.complianceLabels()), prevalence(val.complianceLabels()), prevalence(test.complianceLabels())});
            FigureFactory.figDatasetLabels(taskRates, new File(figDir, "fig1_dataset_labels.png"));

            FigureFactory.figAuroc(meanAuroc, new File(figDir, "fig3_auroc.png"));
            FigureFactory.figOperationalCost(meanCost, new File(figDir, "fig4_operational_cost.png"));

            Map<String, Double> meanTrainTime = new LinkedHashMap<>();
            for (var e : totalTrainTimeByModel.entrySet()) {
                meanTrainTime.put(e.getKey(), e.getValue() / SEEDS.length / 1000.0);
            }
            FigureFactory.figTrainingTime(meanTrainTime, new File(figDir, "fig6_training_time.png"));

            // Task correlation: train labels (fraud, return, compliance)
            String[] taskNames = {"Fraud", "AnyReturn", "Compliance"};
            double[][] corr = correlationMatrix(new int[][]{
                    train.fraudLabels(), train.anyReturnLabels(), train.complianceLabels()});
            FigureFactory.figTaskCorrelation(corr, taskNames, new File(figDir, "fig2_task_correlation.png"));

            // Persist machine-readable results
            Map<String, Object> resultsJson = new LinkedHashMap<>();
            resultsJson.put("config", Map.of(
                    "data_source", cfg.source().name(),
                    "n_transactions", train.size() + val.size() + test.size(),
                    "seeds", SEEDS,
                    "xgboost_rounds", XGBOOST_ROUNDS,
                    "feature_dim", train.featureDim(),
                    "paysim_path", cfg.paysimPath() == null ? "" : cfg.paysimPath().toString()
            ));
            resultsJson.put("auroc_mean", meanAuroc);
            resultsJson.put("operational_cost_mean", meanCost);
            resultsJson.put("training_time_mean_seconds", meanTrainTime);
            resultsJson.put("test_prevalences", taskRates);
            resultsJson.put("task_correlation_matrix", corr);
            resultsJson.put("calibrated_thresholds_by_model", byModelThresholds);
            new ObjectMapper().writerWithDefaultPrettyPrinter()
                    .writeValue(new File(outDir, "results.json"), resultsJson);

            long elapsed = (System.currentTimeMillis() - t0) / 1000;
            System.out.printf("[OK] Pipeline complete in %d seconds. See %s%n", elapsed, outDir.getAbsolutePath());
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Calibrate thresholds on validation, then compute Operational Cost on test.
     * This is the fair-comparison procedure used for the headline cost numbers.
     */
    private static OperationalCost.CostBreakdown costWithCalibration(
            double[] valFraud, double[] valAnyRet, double[] valComp,
            int[] valFraudY, int[] valAnyRetY, int[] valCompY,
            double[] testFraud, double[] testAnyRet, double[] testComp,
            int[] testFraudY, int[] testAnyRetY, int[] testCompY,
            Map<String, ThresholdCalibrator.CalibratedThreshold> outThresholds) {

        ThresholdCalibrator.CalibratedThreshold thrF =
                ThresholdCalibrator.calibrate(valFraud, valFraudY, OperationalCost.FRAUD);
        ThresholdCalibrator.CalibratedThreshold thrA =
                ThresholdCalibrator.calibrate(valAnyRet, valAnyRetY, OperationalCost.ANYRETURN);
        ThresholdCalibrator.CalibratedThreshold thrC =
                ThresholdCalibrator.calibrate(valComp, valCompY, OperationalCost.COMPLIANCE);

        if (outThresholds != null) {
            outThresholds.put("fraud", thrF);
            outThresholds.put("anyReturn", thrA);
            outThresholds.put("compliance", thrC);
        }

        return OperationalCost.compute(
                testFraud,  testFraudY,  thrF.threshold(),
                testAnyRet, testAnyRetY, thrA.threshold(),
                testComp,   testCompY,   thrC.threshold()
        );
    }

    // ------------- helpers -------------

    private record TaskMetrics(Metrics.BinaryMetrics fraud, Metrics.BinaryMetrics anyReturn, Metrics.BinaryMetrics compliance) {}

    private static LabeledDataset standardize(LabeledDataset ds, Standardizer s) {
        return new LabeledDataset(
                s.transform(ds.features()),
                ds.fraudLabels(), ds.returnCodeLabels(), ds.anyReturnLabels(),
                ds.complianceLabels(), ds.transactionIds());
    }

    private static double prevalence(int[] y) {
        int s = 0;
        for (int v : y) s += v;
        return s / (double) y.length;
    }

    private static <T> void record(Map<String, List<T>> map, String k, T v) {
        map.computeIfAbsent(k, x -> new ArrayList<>()).add(v);
    }

    private static void addTime(Map<String, Double> m, String k, long ms) {
        m.merge(k, (double) ms, Double::sum);
    }

    @FunctionalInterface
    private interface DoubleExtractor<T> { double extract(T t); }

    private static <T> double mean(List<T> xs, DoubleExtractor<T> f) {
        double s = 0;
        for (T x : xs) s += f.extract(x);
        return s / xs.size();
    }

    private static double[][] correlationMatrix(int[][] tasks) {
        int k = tasks.length;
        double[][] c = new double[k][k];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                c[i][j] = pearson(tasks[i], tasks[j]);
            }
        }
        return c;
    }

    private static double pearson(int[] a, int[] b) {
        int n = a.length;
        double ma = 0, mb = 0;
        for (int v : a) ma += v;
        for (int v : b) mb += v;
        ma /= n; mb /= n;
        double cov = 0, va = 0, vb = 0;
        for (int i = 0; i < n; i++) {
            double da = a[i] - ma;
            double db = b[i] - mb;
            cov += da * db;
            va += da * da;
            vb += db * db;
        }
        if (va == 0 || vb == 0) return 0;
        return cov / Math.sqrt(va * vb);
    }
}
