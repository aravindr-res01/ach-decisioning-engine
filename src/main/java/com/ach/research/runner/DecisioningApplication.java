package com.ach.research.runner;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Spring Boot entry point for the ACH AI-driven decisioning reference
 * implementation. Two run-modes:
 *
 *   --mode=experiment   Reproduce the empirical evaluation in the paper:
 *                       generate dataset, train all models across seeds,
 *                       compute metrics, render figures, write results JSON.
 *
 *   --mode=serve        Serve the trained MTL-MLP via REST + Kafka (default).
 *                       Used in production-style benchmarking and as the
 *                       reference architecture demonstration.
 *
 * Usage:
 *   mvn spring-boot:run -Dspring-boot.run.arguments="--mode=experiment"
 *   mvn spring-boot:run -Dspring-boot.run.arguments="--mode=serve"
 */
@SpringBootApplication(scanBasePackages = "com.ach.research")
public class DecisioningApplication {

    public static void main(String[] args) {
        String mode = "serve";
        for (String a : args) {
            if (a.startsWith("--mode=")) mode = a.substring("--mode=".length());
        }

        if ("experiment".equalsIgnoreCase(mode)) {
            // Run experiments without Spring's web stack
            ExperimentRunner.RunConfig runCfg = ExperimentRunner.parseArgs(args);
            ExperimentRunner.runFullPipeline(runCfg);
            return;
        }

        // Default: full Spring Boot, REST API, Kafka, etc.
        SpringApplication.run(DecisioningApplication.class, args);
    }
}
