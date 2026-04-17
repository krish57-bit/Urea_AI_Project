#ifndef SIGNAL_PROCESSOR_H
#define SIGNAL_PROCESSOR_H

#include <Arduino.h>

/*
 * SignalProcessor handles the Edge Pre-processing logic:
 * - Interpolation for missing UART packets
 * - Smoothing (Moving Average)
 * - Baseline Subtraction (Environment Cancellation)
 */

class SignalProcessor {
public:
    static const int SCAN_POINTS = 180;
    
    SignalProcessor();
    
    // Core Processing pipeline
    void processScan(float* rawRSSI, float* processedRSSI);
    
    // Environment Calibration
    void captureBaseline(float* rawScan);
    bool hasBaseline() { return _baselineCaptured; }
    
    // Utility functions for TinyML
    void interpolateGaps(float* data);
    void smoothData(float* data, int windowSize);
    
    // Anomaly Detection: Returns 0 (Valid), 1 (Suspect), 2 (Low Confidence Rescan)
    int evaluateInferenceConfidence(float score);

private:
    float _baseline[SCAN_POINTS];
    bool _baselineCaptured = false;
    float _prevValue = -30.0; // Typical RSSI baseline
};

#endif
