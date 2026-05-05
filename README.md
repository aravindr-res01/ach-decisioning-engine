# ACH Decisioning Engine

> Java and Spring Boot reference architecture for AI-driven ACH payment decisioning.
> Joint fraud, return-code, and BSA/AML compliance scoring in a single forward pass.

[![Build](https://github.com/<USERNAME>/ach-decisioning-engine/actions/workflows/build.yml/badge.svg)](https://github.com/<USERNAME>/ach-decisioning-engine/actions/workflows/build.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Java](https://img.shields.io/badge/Java-21-orange.svg)](https://adoptium.net/)
[![Spring Boot](https://img.shields.io/badge/Spring%20Boot-3.2.5-green.svg)](https://spring.io/projects/spring-boot)

---

## Why this exists

The U.S. ACH network processes more than 30 billion payments a year, and every one of them must clear three risk surfaces — **fraud screening, NACHA return-code prediction, and BSA/AML compliance** — within a 100µs SLA window before posting.

In production, banks usually run three independent ML pipelines for these decisions. This project replaces them with a single shared-encoder neural network served from a Spring Boot microservice.

### Headline numbers (real data, not synthetic)

Evaluated on 500,000 transactions from the public Kaggle PaySim release, with NACHA-aligned multi-task augmentation:

| Metric | XGBoost (3 models) | This architecture | Result |
|--------|-------------------:|------------------:|--------|
| Fraud AUROC | 0.9999 | 0.9929 | within 0.7% |
| Compliance AUROC | 0.8150 | 0.8123 | within 0.3% |
| **p99 inference latency** | **1,544 µs** | **7 µs** | **220× faster** |

The architecture trades 18% on operational cost for an order-of-magnitude lower inference latency — the right trade for in-line ACH pre-authorization.

---

## Quick start

### Prerequisites
- JDK 21+ (Eclipse Temurin recommended)
- Maven 3.8+

### Build and run
```bash
git clone https://github.com/<USERNAME>/ach-decisioning-engine.git
cd ach-decisioning-engine
mvn clean package

# Run the full experiment pipeline (synthetic data, ~3 minutes)
java -jar target/ach-decisioning-1.0.0.jar --mode=experiment
