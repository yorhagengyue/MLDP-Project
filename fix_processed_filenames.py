"""
Fix processed filenames to match actual files (clean -> processed references)
"""
import json

notebook_path = 'Geng_Yue__Program_Codes_Submission.ipynb'

# Read notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Fix filenames in all cells
changes = 0
for cell in notebook['cells']:
    if 'source' in cell:
        for i, line in enumerate(cell['source']):
            original = line
            # Fix processed -> clean for data files
            line = line.replace("X_train_processed.csv", "X_train_clean.csv")
            line = line.replace("X_test_processed.csv", "X_test_clean.csv")
            line = line.replace("y_train_log.csv", "train_clean.csv")

            # Fix the y_train loading to use correct column
            if "y_train_log.csv" in original and "SalePrice_log" in original:
                line = line.replace("['SalePrice_log']", "['SalePrice']")

            if line != original:
                cell['source'][i] = line
                changes += 1
                print(f"Fixed: {original.strip()[:100]}")
                print(f"    -> {line.strip()[:100]}")

print(f"\nTotal changes made: {changes}")

# Save fixed notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"[OK] Notebook filenames fixed and saved")
print("\nIMPORTANT: Reload the notebook in VS Code to see changes!")
