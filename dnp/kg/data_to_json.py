# smallcat 231027

import pandas as pd
import json
import sys, getopt
import time

def printUsage():
    print('Usage: python3 nx-compose.py <inputfile1> <inputfile2>')

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h") # opts = [("-h", " ")], args = [inputfile1, inputfile2]
    except getopt.GetoptError:
        printUsage()
        sys.exit(1)
    for opt, arg in opts:
        if opt == '-h':
            printUsage()
            sys.exit()
        else:
            printUsage()
            sys.exit(1)
    if len(args) != 2:
            printUsage()
            sys.exit(1)       

    result = {}

    input_file = "001_out.txt"
    with open(input_file, 'r') as file:
        for line in file:
            line = eval(line)
            len = len(line)
            key = line[0]
            if key not in result:
                result[key] = {}
                splits = line[1].split("-")
                result[key]

            # 确保每行至少有三个元素，以便进行转换
            if len(line) >= 3:
                key1, key2, value = line
                value = int(value)  # 将值转换为整数

                # 构建嵌套的字典结构
                if key1 not in result:
                    result[key1] = {}
                if key2 not in result[key1]:
                    result[key1][key2] = {}
                result[key1][key2][line[1]] = value

    # 保存为JSON文件
    with open('output.json', 'w') as json_file:
        json.dump(result, json_file, indent=4)

    print("JSON文件已保存")


if __name__ == "__main__":
   main(sys.argv[1:])  
