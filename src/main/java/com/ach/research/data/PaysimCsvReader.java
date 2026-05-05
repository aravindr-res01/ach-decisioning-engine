package com.ach.research.data;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Reader for the canonical Kaggle PaySim release
 * (<a href="https://www.kaggle.com/datasets/ealaxi/paysim1">ealaxi/paysim1</a>),
 * 6{,}362{,}620 rows, schema:
 *
 *   step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
 *   nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
 *
 * Note the upstream column-naming inconsistency: oldbalance<b>Org</b> but
 * newbalance<b>Orig</b>. We accept both spellings defensively.
 *
 * Type mapping: PaySim's mobile-money types map to our ACH-aligned
 * {@link Transaction.PaymentType} enum as:
 *   CASH_IN  -> CREDIT
 *   CASH_OUT -> CASH_OUT
 *   DEBIT    -> DEBIT
 *   PAYMENT  -> PAYMENT
 *   TRANSFER -> TRANSFER
 *
 * Use this reader to evaluate the reference architecture against real
 * (publicly released) ACH-style transaction data, in lieu of the
 * synthetic {@link PaySimGenerator}. The {@link NachaAugmentor} should
 * still be run on the output of this reader to add NACHA return-code
 * and BSA/AML compliance labels --- the upstream PaySim dataset only
 * provides the binary fraud label.
 *
 * Kaggle download: see manuscript/README.md or:
 *   pip install kaggle
 *   kaggle datasets download -d ealaxi/paysim1 -p ./data --unzip
 */
public final class PaysimCsvReader {

    private PaysimCsvReader() {}

    /**
     * Read the PaySim CSV in full.
     * <p>
     * Memory cost: ~1.3 GB for 6.36M rows -> increase JVM heap if needed
     * (e.g., {@code -Xmx4g}).
     */
    public static List<Transaction> read(Path csvPath) throws IOException {
        return read(csvPath, Integer.MAX_VALUE, -1L);
    }

    /**
     * Read the PaySim CSV with optional subsampling.
     *
     * @param csvPath    path to PaySim CSV (typically PS_*_log.csv from Kaggle).
     * @param maxRows    upper bound on rows to keep (use {@link Integer#MAX_VALUE}
     *                   for no cap). Subsampling is uniform-random when
     *                   {@code maxRows} is less than the file size.
     * @param seed       RNG seed for subsampling. Pass -1 to take the first
     *                   {@code maxRows} rows in file order (faster, deterministic).
     */
    public static List<Transaction> read(Path csvPath, int maxRows, long seed) throws IOException {
        List<Transaction> out = new ArrayList<>(Math.min(maxRows, 1_000_000));
        RandomGenerator rng = seed >= 0 ? new MersenneTwister(seed) : null;

        // First pass: if random sampling is requested, count rows for proper sampling.
        // For simplicity we use reservoir sampling on a single pass when seed >= 0.
        try (Reader r = new FileReader(csvPath.toFile());
             CSVParser p = CSVFormat.DEFAULT.builder()
                     .setHeader().setSkipHeaderRecord(true).setAllowMissingColumnNames(true).build()
                     .parse(r)) {

            long row = 0;
            for (CSVRecord rec : p) {
                Transaction tx = parseRow(rec, row);
                if (rng == null) {
                    if (out.size() < maxRows) {
                        out.add(tx);
                    } else {
                        break;
                    }
                } else {
                    // Reservoir sampling (algorithm R)
                    if (out.size() < maxRows) {
                        out.add(tx);
                    } else {
                        long j = (long) (rng.nextDouble() * (row + 1));
                        if (j < maxRows) out.set((int) j, tx);
                    }
                }
                row++;
            }
        }
        return out;
    }

    private static Transaction parseRow(CSVRecord rec, long row) {
        int step      = Integer.parseInt(rec.get("step"));
        String type   = rec.get("type").trim().toUpperCase();
        double amount = Double.parseDouble(rec.get("amount"));
        String nameOrig = rec.get("nameOrig");

        // PaySim has a known typo: oldbalanceOrg vs newbalanceOrig.
        // Accept both spellings defensively in case the user re-emits.
        double oldOrig = parseDoubleColumn(rec, "oldbalanceOrg", "oldbalanceOrig");
        double newOrig = parseDoubleColumn(rec, "newbalanceOrig", "newbalanceOrg");
        String nameDest = rec.get("nameDest");
        double oldDest = parseDoubleColumn(rec, "oldbalanceDest");
        double newDest = parseDoubleColumn(rec, "newbalanceDest");
        int isFraud   = Integer.parseInt(rec.get("isFraud"));

        return new Transaction(
                row,
                step,
                mapPaymentType(type),
                amount,
                nameOrig, oldOrig, newOrig,
                nameDest, oldDest, newDest,
                Transaction.ReturnCode.NO_RETURN,    // assigned by NachaAugmentor
                isFraud,
                0                                    // compliance flag assigned by NachaAugmentor
        );
    }

    private static Transaction.PaymentType mapPaymentType(String paysimType) {
        // PaySim uses hyphenated CASH-IN / CASH-OUT in some derivative releases;
        // the canonical Kaggle CSV uses CASH_IN / CASH_OUT (underscore) but we
        // accept both for robustness.
        return switch (paysimType.replace('-', '_')) {
            case "CASH_IN"  -> Transaction.PaymentType.CREDIT;       // semantic alias
            case "CASH_OUT" -> Transaction.PaymentType.CASH_OUT;
            case "DEBIT"    -> Transaction.PaymentType.DEBIT;
            case "TRANSFER" -> Transaction.PaymentType.TRANSFER;
            case "PAYMENT"  -> Transaction.PaymentType.PAYMENT;
            default -> throw new IllegalArgumentException("Unknown PaySim type: " + paysimType);
        };
    }

    private static double parseDoubleColumn(CSVRecord rec, String... candidates) {
        for (String c : candidates) {
            try {
                String v = rec.get(c);
                if (v != null) return Double.parseDouble(v);
            } catch (IllegalArgumentException ignored) {
                // CSVRecord throws if the header isn't found; try next alias.
            }
        }
        throw new IllegalArgumentException("None of the candidate columns found: "
                + String.join(", ", candidates));
    }
}
