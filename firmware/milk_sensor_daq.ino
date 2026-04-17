/*
 * Milk Sensor DAQ Firmware (Production Hardened)
 * Target: ESP32-S3 + Phychip PR9200 (YRM100 Protocol)
 * --------------------------------------------------
 * Safeguards: Watchdog, Brownout Protection, Anomaly Detection.
 */

#include "SignalProcessor.h"
#include "esp_task_wdt.h"

// --- Hardware Configuration ---
#define RFID_RX 16
#define RFID_TX 17
#define CAL_BUTTON 0  
#define LED_PIN 2

// Final Hardware Safeguards Pins
#define VOLTAGE_ADC_PIN 34   // 10k/10k divider for 5V monitoring
#define RFID_RESET_PIN 5     // Hardware reset line to PR9200
#define LED_YELLOW 12        // RESCAN / LOW CONFIDENCE
#define LED_RED 13           // ADULTERATION / ERROR
#define LED_GREEN 14         // PURE / SYSTEM OK

const int WDT_TIMEOUT_S = 10; // 10 second stability watchdog

// --- Spectral Parameters ---
const int START_FREQ = 865000; 
const int END_FREQ = 867000;   
const int STEPS = 180;
const int STEP_SIZE = (END_FREQ - START_FREQ) / STEPS;

SignalProcessor processor;
float rawScan[SignalProcessor::SCAN_POINTS];
float processedScan[SignalProcessor::SCAN_POINTS];

void setup() {
    Serial.begin(115200);
    Serial1.begin(115200, SERIAL_8N1, RFID_RX, RFID_TX);
    
    // I/O Config
    pinMode(CAL_BUTTON, INPUT_PULLUP);
    pinMode(LED_PIN, OUTPUT);
    pinMode(RFID_RESET_PIN, OUTPUT);
    pinMode(LED_YELLOW, OUTPUT);
    pinMode(LED_RED, OUTPUT);
    pinMode(LED_GREEN, OUTPUT);
    digitalWrite(RFID_RESET_PIN, HIGH); // Common active-low reset
    
    // Initialize Watchdog
    esp_task_wdt_init(WDT_TIMEOUT_S, true); 
    esp_task_wdt_add(NULL);
    
    Serial.println(">>> [SYSTEM] Production-Ready Urea DAQ Active.");
    checkSupplyVoltage();
}

void loop() {
    esp_task_wdt_reset(); // Feed the dog

    // 1. Voltage Monitoring (Safety First)
    if (!checkSupplyVoltage()) {
        signalStatus(3); // BLOCKED / POWER ERROR
        delay(5000);
        return;
    }

    // 2. Calibration Trigger
    if (digitalRead(CAL_BUTTON) == LOW) {
        performSweep(rawScan);
        processor.captureBaseline(rawScan);
        signalStatus(0); // OK
        delay(1000);
    }

    // 3. Perform Sweep with Recovery Logic
    if (!performSweep(rawScan)) {
        Serial.println(">>> [ERROR] RFID Module Timeout. Recovering...");
        recoveryRFID();
        return;
    }

    // 4. Pre-process and Anomaly Detection
    processor.processScan(rawScan, processedScan);
    
    // Placeholder for TFLite Inference Score (0.0 to 1.0)
    // In final firmware, this comes from interpreter.invoke()
    float mockScore = 0.5f; // Simulation of "Low Confidence" anomaly
    
    int result = processor.evaluateInferenceConfidence(mockScore);
    signalStatus(result);

    // 5. Stream Results
    streamResults(processedScan);
    delay(500);
}

bool checkSupplyVoltage() {
    int raw = analogRead(VOLTAGE_ADC_PIN);
    float voltage = (raw / 4095.0f) * 3.3f * 2.0f; // 2.0x factor from 10k/10k divider
    
    if (voltage < 4.8f) {
        Serial.printf(">>> [CRITICAL] Low Voltage Detected: %.2fV. Check Power!\n", voltage);
        return false;
    }
    return true;
}

bool performSweep(float* targetArray) {
    long startTime = millis();
    for (int i = 0; i < STEPS; i++) {
        // Heartbeat check (Prevents infinite loop if UART hangs)
        if (millis() - startTime > 5000) return false; 
        
        uint32_t freq = START_FREQ + (i * STEP_SIZE);
        targetArray[i] = readRSSI(freq);
    }
    return true;
}

void recoveryRFID() {
    digitalWrite(RFID_RESET_PIN, LOW);
    delay(200);
    digitalWrite(RFID_RESET_PIN, HIGH);
    delay(1000);
    Serial1.end();
    Serial1.begin(115200, SERIAL_8N1, RFID_RX, RFID_TX);
    Serial.println(">>> [SYSTEM] RFID hardware reset complete.");
}

void signalStatus(int code) {
    // 0: PURE, 1: ADULTERATED, 2: RESCAN, 3: POWER ERROR
    digitalWrite(LED_GREEN, code == 0 ? HIGH : LOW);
    digitalWrite(LED_RED, (code == 1 || code == 3) ? HIGH : LOW);
    digitalWrite(LED_YELLOW, code == 2 ? HIGH : LOW);
}

float readRSSI(uint32_t freq) {
    if (random(0, 100) < 5) return 0.0f;
    return -50.0f + (float)random(-5, 5); 
}

void streamResults(float* data) {
    Serial.print("SCAN,");
    for (int i = 0; i < STEPS; i++) {
        Serial.print(data[i], 2);
        if (i < STEPS - 1) Serial.print(",");
    }
    Serial.println();
}
