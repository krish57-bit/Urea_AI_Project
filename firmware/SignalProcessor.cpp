#include "SignalProcessor.h"

SignalProcessor::SignalProcessor() {
    for (int i = 0; i < SCAN_POINTS; i++) {
        _baseline[i] = 0.0f;
    }
}

void SignalProcessor::captureBaseline(float* rawScan) {
    for (int i = 0; i < SCAN_POINTS; i++) {
        _baseline[i] = rawScan[i];
    }
    _baselineCaptured = true;
    Serial.println(">>> [SYSTEM] Environmental Baseline Captured and Subtracted.");
}

void SignalProcessor::processScan(float* rawRSSI, float* processedRSSI) {
    // 1. Copy raw to processed
    for (int i = 0; i < SCAN_POINTS; i++) {
        processedRSSI[i] = rawRSSI[i];
    }

    // 2. Linear Interpolation for dropped packets (Zeros in UART stream)
    interpolateGaps(processedRSSI);

    // 3. Environmental Subtraction (If calibrated)
    if (_baselineCaptured) {
        for (int i = 0; i < SCAN_POINTS; i++) {
            processedRSSI[i] = processedRSSI[i] - _baseline[i];
        }
    }

    // 4. Smoothing (Moving Average Window of 3)
    smoothData(processedRSSI, 3);
}

void SignalProcessor::interpolateGaps(float* data) {
    for (int i = 1; i < SCAN_POINTS - 1; i++) {
        // If a reading is 0.0 (dropped packet) but neighbors exist
        if (data[i] == 0.0f && data[i-1] != 0.0f && data[i+1] != 0.0f) {
            data[i] = (data[i-1] + data[i+1]) / 2.0f;
        }
    }
}

void SignalProcessor::smoothData(float* data, int windowSize) {
    float smoothed[SCAN_POINTS];
    for (int i = 0; i < SCAN_POINTS; i++) {
        float sum = 0;
        int count = 0;
        for (int j = -(windowSize/2); j <= (windowSize/2); j++) {
            if (i+j >= 0 && i+j < SCAN_POINTS) {
                sum += data[i+j];
                count++;
            }
        }
        smoothed[i] = sum / (float)count;
    }
    
    // Copy back
    for (int i = 0; i < SCAN_POINTS; i++) {
        data[i] = smoothed[i];
    }
}

int SignalProcessor::evaluateInferenceConfidence(float score) {
    // 0.4 to 0.6 is the Anomaly Dead-Zone (HALLUCINATION)
    if (score >= 0.4f && score <= 0.6f) {
        return 2; // Rescan
    }
    
    // Final check for Urea Adulteration
    return (score > 0.6f) ? 1 : 0;
}
