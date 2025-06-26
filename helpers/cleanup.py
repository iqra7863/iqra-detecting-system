import os
import time

def clean_old_screenshots(folder='screenshots', keep_hours=24):
    now = time.time()
    cutoff = now - (keep_hours * 3600)

    if not os.path.exists(folder):
        return

    deleted_files = 0
    for filename in os.listdir(folder):
        if filename.startswith("important_"):  # ⛔ Skip protected files
            continue

        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            file_time = os.path.getmtime(filepath)
            if file_time < cutoff:
                os.remove(filepath)
                deleted_files += 1

    print(f"[🧹 Cleanup] Deleted {deleted_files} old screenshots.")
