package com.ach.research.data;

import com.fasterxml.jackson.annotation.JsonInclude;

/**
 * Canonical ACH transaction record used throughout the decisioning pipeline.
 * Schema follows Lopez-Rojas et al. PaySim with NACHA-aligned augmentation:
 *   - paymentType: ACH-relevant transaction type (DEBIT, CREDIT, CASH_OUT, TRANSFER, PAYMENT)
 *   - amount: principal in account currency
 *   - origBalanceBefore/origBalanceAfter: originating account balance pre/post
 *   - destBalanceBefore/destBalanceAfter: destination account balance pre/post
 *   - step: discrete time index (1 step = 1 hour in PaySim)
 *   - returnCode: NACHA-aligned label, NO_RETURN if successfully posted
 *   - isFraud: ground-truth fraud label
 *   - complianceFlag: BSA/AML rule trigger (structuring, velocity, round-trip, large-unusual)
 *
 * Implemented as a Java record for immutability and concise generation.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record Transaction(
        long transactionId,
        int step,
        PaymentType paymentType,
        double amount,
        String originAccount,
        double origBalanceBefore,
        double origBalanceAfter,
        String destAccount,
        double destBalanceBefore,
        double destBalanceAfter,
        ReturnCode returnCode,
        int isFraud,
        int complianceFlag
) {

    public enum PaymentType {
        DEBIT, CREDIT, CASH_OUT, TRANSFER, PAYMENT;

        public static PaymentType fromOrdinalSafe(int idx) {
            PaymentType[] all = values();
            return all[Math.floorMod(idx, all.length)];
        }
    }

    /**
     * NACHA return codes covered in this study. NO_RETURN denotes a successfully
     * posted transaction. R01-R29 are the most operationally significant return
     * codes per NACHA Operating Rules (representing >95% of returns in practice).
     */
    public enum ReturnCode {
        NO_RETURN,        // 0 - successful posting
        R01,              // Insufficient Funds
        R02,              // Account Closed
        R03,              // No Account / Unable to Locate
        R04,              // Invalid Account Number
        R05,              // Unauthorized Debit (consumer)
        R07,              // Authorization Revoked
        R08,              // Payment Stopped
        R10,              // Customer Advises Originator Not Authorized
        R16,              // Account Frozen
        R20,              // Non-Transaction Account
        R23,              // Credit Entry Refused
        R29;              // Corporate Customer Advises Not Authorized

        public static ReturnCode fromOrdinalSafe(int idx) {
            ReturnCode[] all = values();
            return all[Math.floorMod(idx, all.length)];
        }

        public boolean isReturned() {
            return this != NO_RETURN;
        }
    }

    /** Convenience accessor: any-return label for the multi-task head. */
    public int anyReturnLabel() {
        return returnCode.isReturned() ? 1 : 0;
    }
}
