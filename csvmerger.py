import csv
import glob

combined = []

for filename in glob.glob("./data2/*.csv"):
    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            combined.append(row)

with open("fnn2.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["SCALAR", "ACTIVATION", "EPOCH"])
    writer.writerows(combined)