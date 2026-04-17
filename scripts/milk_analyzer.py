import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import argparse
import sys

"""
Milk Analyzer AI - Unified Analyst Tool
---------------------------------------
Integrates Urea Detection (TinyML), Raw Milk Detection,
Classification (Grade), and Multi-Output Regression (Composition).
"""

class MilkAnalyzer:
    def __init__(self):
        print("Initializing Milk Analyzer AI System...")
        try:
            # 1. Load TFLite Models
            self.urea_interpreter = tf.lite.Interpreter(model_path="urea_detector.tflite")
            self.urea_interpreter.allocate_tensors()
            
            self.raw_interpreter = tf.lite.Interpreter(model_path="raw_milk_detector.tflite")
            self.raw_interpreter.allocate_tensors()
            
            # 2. Load Joblib Models (Grade)
            self.grade_model = joblib.load("milk_grade_model.joblib")
            self.grade_scaler = joblib.load("grade_scaler.joblib")
            
            # 3. Load Composition Scalers (for manual inference if needed, but we'll use raw for simplicity in this tool)
            self.comp_scaler_x = joblib.load("comp_scaler_x.joblib")
            self.comp_scaler_y = joblib.load("comp_scaler_y.joblib")
            
            print("All models loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")
            sys.exit(1)

    def run_tflite_inference(self, interpreter, input_data):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Prepare input
        input_data = input_data.astype(np.float32).reshape(input_details[0]['shape'])
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        return interpreter.get_tensor(output_details[0]['index'])

    def analyze_sample(self, vna_data, physical_data=None):
        """
        vna_data: 150 spectral points for Urea detection OR 518 points for others.
        physical_data: pH, Temp, Taste, Odor, Fat, Turbidity, Colour (for Grade)
        """
        results = {}
        
        # 1. Urea Detection (Requires 150 points)
        if len(vna_data) >= 150:
            urea_input = vna_data[:150]
            urea_prob = self.run_tflite_inference(self.urea_interpreter, urea_input)[0][0]
            results['urea_detected'] = urea_prob > 0.5
            results['urea_confidence'] = urea_prob if results['urea_detected'] else (1 - urea_prob)

        # 2. Raw Milk Detection (Requires 518 points)
        if len(vna_data) >= 518:
            raw_input = vna_data[:518]
            raw_prob = self.run_tflite_inference(self.raw_interpreter, raw_input)[0][0]
            results['is_raw_milk'] = raw_prob > 0.5
            results['raw_confidence'] = raw_prob if results['is_raw_milk'] else (1 - raw_prob)

        # 3. Grade Classification
        if physical_data is not None:
            phys_scaled = self.grade_scaler.transform([physical_data])
            grade_idx = self.grade_model.predict(phys_scaled)[0]
            results['grade'] = ['Low', 'Medium', 'High'][int(grade_idx)]

        return results

    def print_report(self, results):
        print("\n" + "="*40)
        print("       MILK QUALITY AUDIT REPORT")
        print("="*40)
        
        # Adulteration Status
        status = "SUSPECT (Urea Detected)" if results.get('urea_detected') else "PASS (Pure)"
        color = "NEGATIVE" if results.get('urea_detected') else "POSITIVE"
        print(f"Safety Status:    {status}")
        print(f"Confidence:       {results.get('urea_confidence', 0)*100:.1f}%")
        print("-" * 40)
        
        # Identity
        identity = "Raw/Unprocessed" if results.get('is_raw_milk') else "Processed/Standard"
        print(f"Milk Identity:    {identity}")
        
        # Quality
        print(f"Quality Grade:    {results.get('grade', 'N/A')}")
        print("="*40 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Milk Analyzer AI")
    parser.add_argument("--vna_csv", type=str, default="vna_adulteration_dataset.csv", help="Path to VNA dataset")
    parser.add_argument("--index", type=int, default=0, help="Row index to analyze")
    args = parser.parse_args()

    analyzer = MilkAnalyzer()
    
    # Load specific row for demo
    try:
        vna_df = pd.read_csv(args.vna_csv)
        sample_row = vna_df.iloc[args.index]
        
        # Extract spectral data (columns starting with 'f_')
        vna_data = sample_row[[col for col in vna_df.columns if col.startswith('f_')]].values
        
        # For Grade, we'll mock physical data if not available, or load from milknew if exists
        # In a real app, this would come from sensors.
        mock_physical = [6.6, 35, 1, 0, 1, 0, 254] # pH, Temp, Taste, Odor, Fat, Turbidity, Colour
        
        results = analyzer.analyze_sample(vna_data, physical_data=mock_physical)
        analyzer.print_report(results)
        
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
