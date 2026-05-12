"""
Separate 100-epoch results from old 1-epoch and 20-epoch results
"""

def separate_results():
    input_file = "result_long_term_forecast.txt"
    archive_file = "result_archive_1epoch_20epoch.txt"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    epoch_100 = []
    old_epochs = []
    
    # Process in blocks of 3 lines (experiment name, results, blank line)
    i = 0
    while i < len(lines):
        if i >= len(lines):
            break
            
        line1 = lines[i].strip()
        line2 = lines[i+1].strip() if i+1 < len(lines) else ""
        line3 = lines[i+2].strip() if i+2 < len(lines) else ""
        
        # Check if this is a 100-epoch result
        if "baseline_100" in line1 or "PFB_100" in line1 or "Ablation_100" in line1:
            epoch_100.append(lines[i])
            if i+1 < len(lines):
                epoch_100.append(lines[i+1])
            if i+2 < len(lines):
                epoch_100.append(lines[i+2])
        else:
            old_epochs.append(lines[i])
            if i+1 < len(lines):
                old_epochs.append(lines[i+1])
            if i+2 < len(lines):
                old_epochs.append(lines[i+2])
        
        i += 3
    
    # Write archived old results
    with open(archive_file, 'w', encoding='utf-8') as f:
        f.writelines(old_epochs)
    
    # Write only 100-epoch results to main file
    with open(input_file, 'w', encoding='utf-8') as f:
        f.writelines(epoch_100)
    
    print(f"✓ Separated results successfully!")
    print(f"  - 100-epoch results: {len(epoch_100)//3} experiments → {input_file}")
    print(f"  - Old (1/20-epoch): {len(old_epochs)//3} experiments → {archive_file}")

if __name__ == "__main__":
    separate_results()
