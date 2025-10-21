import os
import re
from summaryGen import generate_summary

def collect_files():
    output_dir = "./output_data"
    all_files = os.listdir(output_dir)

    nursery_files = [
        os.path.join(output_dir, f)
        for f in all_files
        if re.match(r"masterData_N[A-Z]+B\d+\.csv", f)
    ]

    tank_files = [
        os.path.join(output_dir, f)
        for f in all_files
        if re.match(r"masterData_T\d+B\d+\.csv", f)
    ]

    return nursery_files, tank_files

if __name__ == "__main__":
    nursery_files, tank_files = collect_files()

    if nursery_files:
        print("[INFO] Generating summary for nurseries...")
        generate_summary(nursery_files, "Nursery")
    else:
        print("[WARN] No nursery files found.")

    if tank_files:
        print("[INFO] Generating summary for tanks...")
        generate_summary(tank_files, "Tank")
    else:
        print("[WARN] No tank files found.")
