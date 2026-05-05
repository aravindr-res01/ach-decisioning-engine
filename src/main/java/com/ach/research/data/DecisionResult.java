package com.ach.research.data;

import com.fasterxml.jackson.annotation.JsonInclude;

/**
 * Output of the AI-augmented decisioning service. Captures all three task
 * predictions plus the final routing recommendation derived from calibrated
 * thresholds and the operational cost matrix.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record DecisionResult(
        long transactionId,
        double fraudProbability,
        double anyReturnProbability,
        double[] returnCodeProbabilities,    // dim 12 (NACHA codes)
        double complianceProbability,
        Recommendation recommendation,
        double inferenceLatencyMicros
) {

    public enum Recommendation {
        APPROVE,
        REVIEW_FRAUD,
        REVIEW_COMPLIANCE,
        REVIEW_RETURN_RISK,
        REJECT
    }

    /**
     * Routing logic. Order matters: REJECT > REVIEW_FRAUD > REVIEW_COMPLIANCE >
     * REVIEW_RETURN_RISK > APPROVE. Thresholds are calibrated per task to align
     * with the operational cost matrix (FN cost asymmetry across tasks).
     */
    public static Recommendation route(
            double fraudProb, double complianceProb, double anyReturnProb,
            double fraudThreshold, double complianceThreshold, double returnThreshold) {
        if (fraudProb >= 0.95 || complianceProb >= 0.95) {
            return Recommendation.REJECT;
        }
        if (fraudProb >= fraudThreshold) {
            return Recommendation.REVIEW_FRAUD;
        }
        if (complianceProb >= complianceThreshold) {
            return Recommendation.REVIEW_COMPLIANCE;
        }
        if (anyReturnProb >= returnThreshold) {
            return Recommendation.REVIEW_RETURN_RISK;
        }
        return Recommendation.APPROVE;
    }
}
