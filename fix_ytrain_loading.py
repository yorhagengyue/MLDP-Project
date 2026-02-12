"""
Fix processed filenames correctly
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

            # Fix the y_train line specifically
            if "y_train = pd.read_csv('data/train_clean.csv')['SalePrice'].values" in line:
                line = line.replace(
                    "y_train = pd.read_csv('data/train_clean.csv')['SalePrice'].values",
                    "y_train = pd.read_csv('data/y_train_clean.csv')['SalePrice_log'].values"
                )

            if line != original:
                cell['source'][i] = line
                changes += 1
                print(f"Fixed: {original.strip()[:100]}")
                print(f"    -> {line.strip()[:100]}")

print(f"\nTotal changes made: {changes}")

# Save fixed notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"[OK] Notebook y_train loading fixed")
print("\nIMPORTANT: Reload the notebook in VS Code!")
