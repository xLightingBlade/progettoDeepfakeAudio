import csv
import os

def sort_data():
    print("sorting")
    directory_path = "new_test_audio_data"
    meta_csv = csv.reader(open("new_test_audio_data\\meta.csv"), delimiter=',')
    rows = list(meta_csv)
    next(meta_csv, None)
    for row in rows:
        os.replace(f"{directory_path}\\{row[0]}",
                   f"{directory_path}\\{'fake' if row[2] == 'spoof' else 'real'}\\{row[0]}")

    print("sorted")

