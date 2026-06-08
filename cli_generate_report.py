import os
from generate_report import generate_pdf

def interactive_report_generator():
    print("==================================================")
    print("         AETHER MANUAL PDF REPORT COMPILER        ")
    print("==================================================")
    
    # 1. Safely retrieve the JSON path
    while True:
        json_path = input("Enter path to AETHER JSON file (e.g., results.json): ").strip()
        # Clean enclosing quotes if the file was dragged and dropped into the terminal
        json_path = json_path.strip("'\"")
        if os.path.exists(json_path):
            break
        print(f"Error: File not found at '{json_path}'. Please verify the path and try again.")
    
    # 2. Retrieve customer credentials
    customer_name = input("Enter Customer Name (e.g., Dr. Sarah Chen): ").strip()
    if not customer_name:
        customer_name = "Independent Researcher"
        
    customer_institution = input("Enter Customer Institution (e.g., MIT): ").strip()
    
    # 3. Generate PDF and launch
    print("\nProcessing raw JSON data and compiling layout...")
    try:
        pdf_path = generate_pdf(
            json_input=json_path,
            customer_name=customer_name,
            customer_institution=customer_institution
        )
        print("\n" + "="*50)
        print(f"[SUCCESS] PDF Report generated successfully:")
        print(f"          {pdf_path}")
        print("==================================================")
        
        # Automatically launch the generated PDF on Windows
        try:
            os.startfile(pdf_path)
            print("Launching PDF viewer...")
        except Exception:
            pass
            
    except Exception as e:
        print(f"\n[ERROR] Failed to compile PDF: {e}")

if __name__ == "__main__":
    interactive_report_generator()
