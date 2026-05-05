package com.ach.research.service;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * Externalized decisioning thresholds. Each threshold is calibrated on the
 * validation split to minimize Operational Cost (per-task asymmetric).
 *
 * Defaults below are illustrative. Production values should be re-calibrated
 * monthly against fresh validation data and approved through your bank's
 * model-risk-management process.
 */
@Component
@ConfigurationProperties(prefix = "ach.decisioning")
public class DecisioningConfig {

    private double fraudThreshold = 0.15;
    private double complianceThreshold = 0.20;
    private double anyReturnThreshold = 0.30;
    private String modelArtifactPath = "model/mtl-mlp.json";
    private String scalerPath = "model/scaler.json";

    public double fraudThreshold() { return fraudThreshold; }
    public void setFraudThreshold(double v) { this.fraudThreshold = v; }

    public double complianceThreshold() { return complianceThreshold; }
    public void setComplianceThreshold(double v) { this.complianceThreshold = v; }

    public double anyReturnThreshold() { return anyReturnThreshold; }
    public void setAnyReturnThreshold(double v) { this.anyReturnThreshold = v; }

    public String modelArtifactPath() { return modelArtifactPath; }
    public void setModelArtifactPath(String v) { this.modelArtifactPath = v; }

    public String scalerPath() { return scalerPath; }
    public void setScalerPath(String v) { this.scalerPath = v; }
}
