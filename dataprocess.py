import csv

import pandas as pd

input_file = ["./data/811_rope.csv", "./data/cosine_rope.csv", "./data/wsd_rope.csv"]
output_file = ["./data/811", "./data/cosine", "./data/wsd"]

for (input_file, output_file) in zip(input_file, output_file):
    # 打开输入文件并读取前一万行
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = list(csv.reader(infile))
        first_10000_rows = reader[:10000]

    selected_rows = first_10000_rows[::50]
    all_rows = reader[::50]

    with open(output_file + "_10000.csv", 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(selected_rows)

    with open(output_file + "_full.csv", 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(all_rows)

    print(f"Processed rows saved")

input_file = ["./data/811_10000.csv", "./data/cosine_10000.csv", "./data/wsd_10000.csv", "./data/811_full.csv",
              "./data/cosine_full.csv", "./data/wsd_full.csv"]
output_file = input_file.copy()
for (input_file, output_file) in zip(input_file, output_file):
    # 读取 CSV 文件
    df = pd.read_csv(input_file)

    # 交换第二列和第三列
    columns = list(df.columns)
    columns[1], columns[2] = columns[2], columns[1]
    df = df[columns]

    # 保存到新的 CSV 文件
    df.to_csv(output_file, index=False)

    print(f"Swapped columns saved to {output_file}")
