import re
import os
from collections import Counter

log_path = r"c:\Users\TAOLIVE\OneDrive - Daimler Truck\Desktop\aprendiz\Bot Shield\Bot Shield\logs\detailed_phases\TrendSpecialist_Phase2_Hyperparams_20260306-085537.log"

if not os.path.exists(log_path):
    print(f"Log not found: {log_path}")
else:
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
        
    print(f"Total lines in log: {len(lines)}")
    
    trials_start = len([l for l in lines if "TRIAL" in l and "INICIO" in l])
    evaluates = len([l for l in lines if "EVALUATE end" in l])
    errors = [l.split("ERROR - ")[1].strip() for l in lines if "ERROR - " in l]
    warnings = [l.split("WARNING - ")[1].strip() for l in lines if "WARNING - " in l]
    
    print(f"Trials start count: {trials_start}")
    print(f"Evaluations completed: {evaluates}")
    
    if errors:
        print(f"\nTop Errors ({len(errors)} total):")
        for err, count in Counter(errors).most_common(5):
            print(f"- {count}x: {err[:150]}")
    
    if warnings:
        warn_counter = Counter(warnings)
        print(f"\nTop Warnings ({len(warnings)} total):")
        for w, count in warn_counter.most_common(5):
            print(f"- {count}x: {w[:150]}")
            
    print("\nLast 10 lines:")
    for l in lines[-10:]:
        print(l.strip()[:100])
