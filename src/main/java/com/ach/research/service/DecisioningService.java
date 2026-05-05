package com.ach.research.service;

import com.ach.research.data.DecisionResult;
import com.ach.research.data.FeatureEngineer;
import com.ach.research.data.FeatureVector;
import com.ach.research.data.Standardizer;
import com.ach.research.data.Transaction;
import com.ach.research.model.MultiTaskMlp;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Orchestrates the per-transaction decisioning request:
 *
 *   transaction -> feature engineering -> standardization -> MTL inference
 *               -> threshold-based routing -> {@link DecisionResult}.
 *
 * Single-pass design: ALL three task probabilities and the 12-class return-
 * code distribution are produced in one MTL forward call, which is the
 * architectural property the paper measures and reports.
 */
@Service
public class DecisioningService {

    private final MultiTaskMlp model;
    private final Standardizer scaler;
    private final DecisioningConfig config;
    private final AtomicLong requestCounter = new AtomicLong();

    @Autowired
    public DecisioningService(MultiTaskMlp model, Standardizer scaler, DecisioningConfig config) {
        this.model = model;
        this.scaler = scaler;
        this.config = config;
    }

    public DecisionResult decide(Transaction tx) {
        long t0 = System.nanoTime();

        FeatureVector raw = FeatureEngineer.engineer(tx);
        double[] scaled = scaler.transform(raw.features());
        MultiTaskMlp.Predictions p = model.predictSingle(scaled);

        double fraudProb = p.fraud()[0];
        double anyRetProb = p.anyReturn()[0];
        double compProb = p.compliance()[0];

        DecisionResult.Recommendation rec = DecisionResult.route(
                fraudProb, compProb, anyRetProb,
                config.fraudThreshold(),
                config.complianceThreshold(),
                config.anyReturnThreshold()
        );

        double latencyUs = (System.nanoTime() - t0) / 1000.0;
        requestCounter.incrementAndGet();

        return new DecisionResult(
                tx.transactionId(),
                fraudProb, anyRetProb,
                p.returnCode()[0],
                compProb, rec, latencyUs
        );
    }

    public long requestCount() { return requestCounter.get(); }
}
