"""
Fix all file paths in notebook: '../data/' -> 'data/' and '../models/' -> 'models/'
"""
import json

notebook_path = 'Geng_Yue__Program_Codes_Submission.ipynb'

# Read notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Fix paths in all cells
changes = 0
for cell in notebook['cells']:
    if 'source' in cell:
        for i, line in enumerate(cell['source']):
            original = line
            # Fix data paths
            line = line.replace("'../data/", "'data/")
            line = line.replace('"../data/', '"data/')
            # Fix model paths
            line = line.replace("'../models/", "'models/")
            line = line.replace('"../models/', '"models/')

            if line != original:
                cell['source'][i] = line
                changes += 1
                print(f"Fixed: {original.strip()[:80]}")

print(f"\nTotal changes made: {changes}")

# Save fixed notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"[OK] Notebook paths fixed and saved to {notebook_path}")
print("\nIMPORTANT: Please reload the notebook in VS Code/Jupyter to see the changes!")
