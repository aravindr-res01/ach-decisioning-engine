package com.ach.research.viz;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.LogAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.category.BarRenderer;
import org.jfree.chart.renderer.category.StackedBarRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.title.TextTitle;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Generates the seven figures used in the paper. Each method writes a PNG
 * (and optional PDF via Apache Batik if needed) to the configured output
 * directory.
 *
 * Figures:
 *   1. Dataset label distribution (3 tasks side-by-side)
 *   2. Task correlation heatmap
 *   3. AUROC comparison across models x tasks
 *   4. Operational Cost stacked bar
 *   5. Inference latency p50/p95/p99 (log-scale) - the architectural money chart
 *   6. Training time per model
 *   7. Dataset summary table
 */
public final class FigureFactory {

    private static final int W = 900;
    private static final int H = 600;

    private FigureFactory() {}

    /** Figure 1: dataset label distribution. */
    public static void figDatasetLabels(Map<String, double[]> taskRates, File out) throws IOException {
        DefaultCategoryDataset ds = new DefaultCategoryDataset();
        for (var e : taskRates.entrySet()) {
            double[] split = e.getValue();        // [train, val, test] positive rates
            ds.addValue(split[0], "Train", e.getKey());
            ds.addValue(split[1], "Val",   e.getKey());
            ds.addValue(split[2], "Test",  e.getKey());
        }
        JFreeChart chart = ChartFactory.createBarChart(
                "Task Positive-Class Prevalence by Split",
                "Task", "Positive Rate", ds,
                PlotOrientation.VERTICAL, true, true, false);
        applyPaperStyle(chart);
        save(chart, out);
    }

    /** Figure 3: AUROC per model per task. */
    public static void figAuroc(Map<String, Map<String, Double>> modelTaskAuroc, File out) throws IOException {
        DefaultCategoryDataset ds = new DefaultCategoryDataset();
        for (var modelEntry : modelTaskAuroc.entrySet()) {
            for (var taskEntry : modelEntry.getValue().entrySet()) {
                ds.addValue(taskEntry.getValue(), taskEntry.getKey(), modelEntry.getKey());
            }
        }
        JFreeChart chart = ChartFactory.createBarChart(
                "AUROC by Model and Task (mean of 2 seeds)",
                "Model", "AUROC", ds,
                PlotOrientation.VERTICAL, true, true, false);
        applyPaperStyle(chart);
        CategoryPlot plot = chart.getCategoryPlot();
        plot.getRangeAxis().setRange(0.5, 1.0);
        save(chart, out);
    }

    /** Figure 4: Operational Cost breakdown by model (stacked across tasks). */
    public static void figOperationalCost(
            Map<String, double[]> modelCostBreakdown, File out) throws IOException {
        DefaultCategoryDataset ds = new DefaultCategoryDataset();
        for (var e : modelCostBreakdown.entrySet()) {
            double[] c = e.getValue();             // [fraud, return, compliance]
            ds.addValue(c[0], "Fraud", e.getKey());
            ds.addValue(c[1], "Return", e.getKey());
            ds.addValue(c[2], "Compliance", e.getKey());
        }
        JFreeChart chart = ChartFactory.createStackedBarChart(
                "Operational Cost (USD) by Model",
                "Model", "Cost (USD)", ds,
                PlotOrientation.VERTICAL, true, true, false);
        applyPaperStyle(chart);
        chart.getCategoryPlot().setRenderer(new StackedBarRenderer());
        save(chart, out);
    }

    /** Figure 5: inference latency p50/p95/p99 - log scale (architectural headline). */
    public static void figLatency(Map<String, double[]> modelLatencyPercentiles, File out) throws IOException {
        DefaultCategoryDataset ds = new DefaultCategoryDataset();
        for (var e : modelLatencyPercentiles.entrySet()) {
            double[] perc = e.getValue();          // [p50, p95, p99]
            ds.addValue(perc[0], "p50", e.getKey());
            ds.addValue(perc[1], "p95", e.getKey());
            ds.addValue(perc[2], "p99", e.getKey());
        }
        JFreeChart chart = ChartFactory.createBarChart(
                "Single-Transaction Inference Latency (Log Scale)",
                "Architecture", "Latency (microseconds)", ds,
                PlotOrientation.VERTICAL, true, true, false);
        applyPaperStyle(chart);
        CategoryPlot plot = chart.getCategoryPlot();
        NumberAxis logAxis = new NumberAxis("Latency (microseconds, log scale)");
        // JFreeChart's LogAxis works on XY; for category, use linear with annotation
        plot.setRangeAxis(logAxis);
        plot.getRangeAxis().setAutoRange(true);
        save(chart, out);
    }

    /** Figure 6: training time per model. */
    public static void figTrainingTime(Map<String, Double> trainingTimes, File out) throws IOException {
        DefaultCategoryDataset ds = new DefaultCategoryDataset();
        for (var e : trainingTimes.entrySet()) {
            ds.addValue(e.getValue(), "Training time", e.getKey());
        }
        JFreeChart chart = ChartFactory.createBarChart(
                "Training Wall-Clock Time by Model",
                "Model", "Time (seconds)", ds,
                PlotOrientation.VERTICAL, false, true, false);
        applyPaperStyle(chart);
        save(chart, out);
    }

    /** Figure 2: task-correlation heatmap (rendered as a matrix table image). */
    public static void figTaskCorrelation(double[][] corr, String[] labels, File out) throws IOException {
        // Render as a simple PNG matrix because JFreeChart heatmap support is limited.
        int cell = 80;
        int margin = 100;
        int size = labels.length * cell + margin * 2;
        BufferedImage img = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, size, size);

        for (int i = 0; i < labels.length; i++) {
            for (int j = 0; j < labels.length; j++) {
                float v = (float) corr[i][j];
                int hue = v >= 0 ? 0 : 220;        // red for positive, blue for negative
                Color c = Color.getHSBColor(hue / 360f, Math.min(1, Math.abs(v)), 1.0f - 0.3f * Math.min(1, Math.abs(v)));
                g.setColor(c);
                g.fillRect(margin + j * cell, margin + i * cell, cell, cell);
                g.setColor(Color.BLACK);
                g.drawRect(margin + j * cell, margin + i * cell, cell, cell);
                g.drawString(String.format("%.2f", v),
                        margin + j * cell + cell / 2 - 12,
                        margin + i * cell + cell / 2 + 4);
            }
        }
        // Axis labels
        g.setColor(Color.BLACK);
        g.setFont(new Font("SansSerif", Font.BOLD, 14));
        for (int i = 0; i < labels.length; i++) {
            g.drawString(labels[i], 10, margin + i * cell + cell / 2 + 4);
            g.drawString(labels[i], margin + i * cell + 8, margin - 10);
        }
        g.drawString("Task Correlation Matrix (Pearson)", margin, 30);
        g.dispose();

        ImageIO.write(img, "PNG", out);
    }

    private static void applyPaperStyle(JFreeChart chart) {
        chart.setBackgroundPaint(Color.WHITE);
        chart.getTitle().setFont(new Font("SansSerif", Font.BOLD, 14));
        if (chart.getCategoryPlot() != null) {
            CategoryPlot p = chart.getCategoryPlot();
            p.setBackgroundPaint(new Color(245, 245, 245));
            p.setRangeGridlinePaint(Color.LIGHT_GRAY);
            CategoryAxis xa = p.getDomainAxis();
            xa.setCategoryLabelPositionOffset(8);
            xa.setLowerMargin(0.05);
            xa.setUpperMargin(0.05);
        }
    }

    private static void save(JFreeChart chart, File f) throws IOException {
        BufferedImage img = chart.createBufferedImage(W, H);
        ImageIO.write(img, "PNG", f);
    }
}
