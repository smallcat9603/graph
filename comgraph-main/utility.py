import os
from typing import *


def search_descendant(before: str, after: str, dir: str = "."):
    for f in os.scandir(dir):
        if f.is_file() and before in f.path:
            newname = f.path.replace(before, after)
            os.rename(f.path, newname)
        elif f.is_dir():
            print("f.path: ", f.path)
            if "/." in f.path:
                continue
            search_descendant(before, after, f.path)
            if before in f.path:
                newname = f.path.replace(before, after)
                os.rename(f.path, newname)


def export_table(records: List[List], labels: List[str]):
    lengths = []
    for label in labels:
        lengths.append(len(label))
    for record in records:
        for i, val in enumerate(record):
            lengths[i] = max(lengths[i], len(f"{val}"))
    for record in records:
        if len(record) != len(labels):
            print("all records and labels need to have the same size")
            exit()
    print("|", end="")
    for i, label in enumerate(labels):
        print(f" {label.ljust(lengths[i])} |", end='')
    print()
    print("|", end="")
    for i, _ in enumerate(labels):
        print(f":{'-' * lengths[i]}:|", end='')
    print()
    for record in records:
        print("|", end="")
        for i, val in enumerate(record):
            print(f" {str(val).ljust(lengths[i])} |", end='')
        print()


def export_to_simple_file(records: List[List[float]], path: str = "tmp/tmp.txt"):
    f = open(path, "w")
    for record in records:
        if len(record) < 1:
            print("each record must be longer than 0")
        line = f'{record[0]}'
        for val in record[1:]:
            line += f' {val}'
        line += "\n"
        f.write(line)
    f.close()
    print(path)


def export_to_csv(records: List[List[float]], labels: List[str], path: str = "tmp/tmp.csv"):
    f = open(path, "w")
    line = f"{labels[0]}"
    for label in labels[1:]:
        line += f",{label}"
    line += "\n"
    f.write(line)
    for record in records:
        if len(record) < 1:
            print("each record must be longer than 0")
        line = f'{record[0]}'
        for val in record[1:]:
            line += f',{val}'
        line += "\n"
        f.write(line)
    f.close()
    print(path)


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def represents_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_from_simple_file(path: str = "tmp/tmp.txt") -> List[List]:
    data = []
    f = open(path)
    for line in f.readlines():
        record = line.split()
        for i, v in enumerate(record):
            if represents_int(v):
                record[i] = int(v)
            elif represents_float(v):
                record[i] = float(v)
        data.append(record)
    f.close()
    return data


def convert_comma_colon_string_to_table(path: str):
    f = open(path)
    labels = []
    records = []
    for line in f.readlines():
        elements = line.split(", ")
        elements[-1] = elements[-1].replace("\n", "")
        record = []
        for elem in elements:
            k, v = elem.split(": ")
            if len(elements) > len(labels):
                labels.append(k)
            record.append(v)
        records.append(record)
    export_table(records, labels)


if __name__ == "__main__":
    # convert_comma_colon_string_to_table("tmp/tmp.txt")
    search_descendant("rattus", "Rattus")
