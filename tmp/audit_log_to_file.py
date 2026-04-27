import re
import os
from collections import Counter

log_path = r"c:\Users\TAOLIVE\OneDrive - Daimler Truck\Desktop\aprendiz\Bot Shield\Bot Shield\logs\detailed_phases\TrendSpecialist_Phase2_Hyperparams_20260306-085537.log"
out_path = r"c:\Users\TAOLIVE\OneDrive - Daimler Truck\Desktop\aprendiz\Bot Shield\Bot Shield\tmp\audit_result.txt"

if not os.path.exists(log_path):
    with open(out_path, 'w') as f:
        f.write(f"Log not found: {log_path}\n")
else:
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
        
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"Total lines in log: {len(lines)}\n")
        
        trials_start = [l for l in lines if "TRIAL" in l and "INICIO" in l]
        evaluates = [l for l in lines if "EVALUATE end" in l]
        errors = [l.split("ERROR - ")[1].strip() for l in lines if "ERROR - " in l]
        warnings = [l.split("WARNING - ")[1].strip() for l in lines if "WARNING - " in l]
        pruneds = [l for l in lines if "pruned" in l.lower() or "prune" in l.lower() or "PRUNED" in l]
        
        f.write(f"Trials start count: {len(trials_start)}\n")
        f.write(f"Evaluations completed: {len(evaluates)}\n")
        f.write(f"Trials pruned/pruning evals: {len(pruneds)}\n")
        
        if len(evaluates) > 0:
            f.write("\nEval outputs:\n")
            for e in evaluates:
                f.write(e.strip() + "\n")
                
        if len(pruneds) > 0:
            f.write("\nPruned outputs (last 10):\n")
            for p in pruneds[-10:]:
                f.write(p.strip() + "\n")
        
        if errors:
            f.write(f"\nTop Errors ({len(errors)} total):\n")
            for err, count in Counter(errors).most_common():
                f.write(f"- {count}x: {err}\n")
        else:
            f.write("\nNO ERRORS FOUND.\n")
        
        if warnings:
            warn_counter = Counter(warnings)
            f.write(f"\nTop Warnings ({len(warnings)} total):\n")
            for w, count in warn_counter.most_common(5):
                f.write(f"- {count}x: {w[:200]}\n")
                
        f.write("\nLast 10 lines:\n")
        for l in lines[-10:]:
            f.write(l.strip() + "\n")
