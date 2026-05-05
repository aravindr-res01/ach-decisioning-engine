package com.ach.research.service;

import com.ach.research.data.DecisionResult;
import com.ach.research.data.Transaction;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

/**
 * REST entry-point for the decisioning service. This is the synchronous
 * API surface; the asynchronous path is the Kafka consumer.
 *
 *   POST /api/v1/decide       -> single-transaction decisioning
 *   GET  /api/v1/health       -> liveness + request counter
 *   GET  /api/v1/metrics      -> exposed Prometheus-friendly metrics
 *
 * Per-request latency is captured in {@link DecisionResult#inferenceLatencyMicros}
 * and aggregated across the JVM by the service for offline analysis.
 */
@RestController
@RequestMapping("/api/v1")
public class DecisioningController {

    private final DecisioningService service;

    @Autowired
    public DecisioningController(DecisioningService service) {
        this.service = service;
    }

    @PostMapping("/decide")
    public ResponseEntity<DecisionResult> decide(@RequestBody Transaction tx) {
        DecisionResult result = service.decide(tx);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/health")
    public Map<String, Object> health() {
        Map<String, Object> m = new HashMap<>();
        m.put("status", "UP");
        m.put("requestCount", service.requestCount());
        return m;
    }
}
