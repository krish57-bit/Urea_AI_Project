import serial
import serial.tools.list_ports
import pandas as pd
import time
import os
import argparse

"""
ESP32 Urea Sensor - Production Data Logger
------------------------------------------
Role: Senior AI Data Scientist (Co-Founder)
Purpose: Capturing real hardware noise from ESP32 for production retraining.
Supports "Drop-and-Swap" protocol.
"""

def find_esp32_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "Silicon Labs" in port.description or "USB Serial" in port.description:
            return port.device
    return None

def main():
    parser = argparse.ArgumentParser(description="Urea Sensor HIL Data Logger")
    parser.add_argument("--mock", action="store_true", help="Run in simulation mode (no hardware)")
    args = parser.parse_args()

    output_file = "production_vna_dataset.csv"
    
    print("========================================")
    print("      UREA SENSOR DATA ACQUISITION")
    print("========================================")
    
    ser = None
    if not args.mock:
        port = find_esp32_port()
        if not port:
            print("Error: ESP32-S3 not detected. Ensure it's plugged in or use --mock.")
            return
        ser = serial.Serial(port, 115200, timeout=2)
        print(f"Connected to ESP32 on {port}")
    else:
        print("RUNNING IN MOCK MODE (Simulated Hardware)")

    while True:
        print("\n--- NEW SAMPLE ACQUISITION ---")
        try:
            urea_pct = input("Enter Urea Percentage (0, 1, 3, 5) or 'q' to quit: ")
            if urea_pct.lower() == 'q': break
            
            bottle_id = input("Enter Bottle ID (e.g., B_01): ")
            
            print(f">>> Place {bottle_id} ({urea_pct}%) in 3mm jig and wait for scan...")
            
            # Wait for "SCAN,..." line from Serial
            raw_data = ""
            if args.mock:
                time.sleep(1)
                # Mock a 180-point scan
                mock_points = [str(round(-50.0 + (i*0.1), 2)) for i in range(180)]
                raw_data = "SCAN," + ",".join(mock_points)
            else:
                while True:
                    line = ser.readline().decode('utf-8').strip()
                    if line.startswith("SCAN"):
                        raw_data = line
                        break
            
            # Process and Save
            parts = raw_data.split(",")
            if len(parts) < 181:
                print(f"Warning: Corrupt scan received ({len(parts)-1}/180 points). Retrying...")
                continue
                
            spectral_data = [float(x) for x in parts[1:181]]
            
            # Structure Data
            row = {
                'Timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'Bottle_ID': bottle_id,
                'Urea_Pct': float(urea_pct),
                'Label': 0 if float(urea_pct) == 0 else 1
            }
            # Add freq columns (placeholder names, matching physics sim)
            for i, val in enumerate(spectral_data):
                row[f'f_{i}'] = val
                
            # Append to CSV
            df_new = pd.DataFrame([row])
            if not os.path.exists(output_file):
                df_new.to_csv(output_file, index=False)
            else:
                df_new.to_csv(output_file, mode='a', header=False, index=False)
                
            print(f"Successfully logged sample {bottle_id} to {output_file}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            break

    if ser: ser.close()
    print("\nData collection session ended.")

if __name__ == "__main__":
    main()
