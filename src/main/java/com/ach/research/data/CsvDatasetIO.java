package com.ach.research.data;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * CSV serialization for the engineered dataset. CSV (gzip-friendly via
 * GZIPOutputStream) is chosen over Parquet for this reference implementation
 * to avoid the ~50MB Apache Parquet Java dependency footprint, which is
 * unfavorable in regulated FinTech deployments.
 *
 * For very large datasets (>10M rows) the user should swap in
 * org.apache.parquet:parquet-avro.
 */
public final class CsvDatasetIO {

    private CsvDatasetIO() {}

    public static void write(LabeledDataset ds, File out) throws IOException {
        try (Writer w = new FileWriter(out);
             CSVPrinter p = new CSVPrinter(w, CSVFormat.DEFAULT.builder()
                     .setHeader(buildHeader(ds.featureDim())).build())) {
            for (int i = 0; i < ds.size(); i++) {
                List<Object> row = new ArrayList<>(ds.featureDim() + 5);
                row.add(ds.transactionIds()[i]);
                for (int j = 0; j < ds.featureDim(); j++) {
                    row.add(ds.features()[i][j]);
                }
                row.add(ds.fraudLabels()[i]);
                row.add(ds.returnCodeLabels()[i]);
                row.add(ds.anyReturnLabels()[i]);
                row.add(ds.complianceLabels()[i]);
                p.printRecord(row);
            }
        }
    }

    public static LabeledDataset read(File in) throws IOException {
        try (Reader r = new FileReader(in);
             CSVParser p = CSVFormat.DEFAULT.builder()
                     .setHeader().setSkipHeaderRecord(true).build().parse(r)) {

            List<CSVRecord> records = p.getRecords();
            int n = records.size();
            int dim = records.get(0).size() - 5;

            double[][] features = new double[n][dim];
            int[] fraud = new int[n], rc = new int[n], ar = new int[n], cp = new int[n];
            long[] tids = new long[n];

            for (int i = 0; i < n; i++) {
                CSVRecord rec = records.get(i);
                tids[i] = Long.parseLong(rec.get(0));
                for (int j = 0; j < dim; j++) {
                    features[i][j] = Double.parseDouble(rec.get(1 + j));
                }
                fraud[i] = Integer.parseInt(rec.get(1 + dim));
                rc[i]    = Integer.parseInt(rec.get(2 + dim));
                ar[i]    = Integer.parseInt(rec.get(3 + dim));
                cp[i]    = Integer.parseInt(rec.get(4 + dim));
            }
            return new LabeledDataset(features, fraud, rc, ar, cp, tids);
        }
    }

    private static String[] buildHeader(int dim) {
        String[] h = new String[dim + 5];
        h[0] = "transactionId";
        for (int j = 0; j < dim; j++) h[1 + j] = "f" + j;
        h[1 + dim] = "isFraud";
        h[2 + dim] = "returnCode";
        h[3 + dim] = "anyReturn";
        h[4 + dim] = "complianceFlag";
        return h;
    }
}
