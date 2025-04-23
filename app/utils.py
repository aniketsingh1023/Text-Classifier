import json
import os

def save_to_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
def append_to_db(entry, db_path="data/outputs/classification_db.json"):
    if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
        save_to_json([], db_path)

    with open(db_path, "r+", encoding="utf-8") as db_file:
        try:
            data = json.load(db_file)
        except json.JSONDecodeError:
            data = []

        data.append(entry)
        db_file.seek(0)
        json.dump(data, db_file, indent=4, ensure_ascii=False)
