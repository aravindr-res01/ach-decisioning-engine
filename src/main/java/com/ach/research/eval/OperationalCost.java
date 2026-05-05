package com.ach.research.eval;

/**
 * Operational Cost metric: a domain-specific evaluator that captures the
 * asymmetric financial impact of false negatives vs false positives across
 * the three decisioning tasks. Defined as
 *
 *   OC = sum_t [ FN_t * c_FN_t  +  FP_t * c_FP_t ]
 *
 * with the cost matrix calibrated to industry estimates of recovery cost,
 * BSA/AML penalty exposure, and operational review burden:
 *
 *   Task           c_FN     c_FP     Source/justification
 *   ------------   -----    -----    ---------------------------------------
 *   fraud          $2500    $15      Avg ACH fraud loss / customer-care touch
 *   anyReturn      $25      $5       NSF / NOC handling cost
 *   compliance     $5000    $50      BSA/AML expected per-incident cost
 *
 * This metric is the headline reporting number in the empirical evaluation
 * because traditional rank-based metrics (AUROC) under-represent the
 * asymmetry between fraud miss and friction.
 */
public final class OperationalCost {

    public record CostMatrix(double fnCost, double fpCost) {}
    public record CostBreakdown(double fraudCost, double anyReturnCost, double complianceCost) {
        public double total() { return fraudCost + anyReturnCost + complianceCost; }
    }

    public static final CostMatrix FRAUD     = new CostMatrix(2500, 15);
    public static final CostMatrix ANYRETURN = new CostMatrix(25,   5);
    public static final CostMatrix COMPLIANCE= new CostMatrix(5000, 50);

    private OperationalCost() {}

    public static CostBreakdown compute(
            double[] pFraud, int[] yFraud, double thrFraud,
            double[] pAnyRet, int[] yAnyRet, double thrAnyRet,
            double[] pComp, int[] yComp, double thrComp) {

        return new CostBreakdown(
                taskCost(pFraud, yFraud, thrFraud, FRAUD),
                taskCost(pAnyRet, yAnyRet, thrAnyRet, ANYRETURN),
                taskCost(pComp, yComp, thrComp, COMPLIANCE)
        );
    }

    private static double taskCost(double[] p, int[] y, double thr, CostMatrix cm) {
        double cost = 0;
        for (int i = 0; i < p.length; i++) {
            int pred = p[i] >= thr ? 1 : 0;
            if (pred == 0 && y[i] == 1) cost += cm.fnCost();
            else if (pred == 1 && y[i] == 0) cost += cm.fpCost();
        }
        return cost;
    }
}
