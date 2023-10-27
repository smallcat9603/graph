# smallcat 231027
#
# transform data in data/ to json format

import pandas as pd
import json
import sys, getopt, os

def printUsage():
    print('Usage: python3 data_to_json.py')

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h") # opts = [("-h", " ")], args = []
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
    if len(args) > 0:
            printUsage()
            sys.exit(1)       

    files = os.listdir("data")
    files_with_out = [file for file in files if "_out.txt" in file]
    files_with_tfidf = [file for file in files if "_tfidf.txt" in file]
    
    result = {}

    for input_file in files_with_out:
        with open(input_file, 'r') as file:
            for line in file:
                line = eval(line)
                newKey = False
                if line[0] not in result:
                    elements = line[1].split("-")
                    result[line[0]] = {elements[0]:{elements[1]:{"lines": [elements[2]]}}}  
                    newKey = True              
                for i in range(1, len(line)):
                    if i == 1 and newKey == True:
                        continue
                    elements = line[i].split("-")
                    if elements[0] not in result[line[0]]:
                        result[line[0]][elements[0]] = {elements[1]:{"lines": [elements[2]]}}
                    elif elements[1] not in result[line[0]][elements[0]]:
                        result[line[0]][elements[0]][elements[1]] = {"lines": [elements[2]]}
                    elif elements[2] not in result[line[0]][elements[0]][elements[1]]["lines"]:
                        result[line[0]][elements[0]][elements[1]]["lines"].append(elements[2])

    for input_file in files_with_tfidf:
        with open(input_file, 'r') as file: 
            ref = []
            for line in file:
                line = eval(line)
                if line[0] not in result:
                    ref = line
                else:
                    for i in (1, len(line)):
                        if float(line[i]) > 0:
                            element0 = ref[i].split("-")[0]
                            element1 = ref[i].split("-")[1].split(".")[0]
                            if "tfidf" not in result[line[0]][element0][element1]:
                                result[line[0]][element0][element1]["tfidf"] = float(line[i])
                            
    with open('data.json', 'w') as json_file:
        json.dump(result, json_file, indent=4)

    print("saved in data.json")


if __name__ == "__main__":
   main(sys.argv[1:])  
