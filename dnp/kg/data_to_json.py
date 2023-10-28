# smallcat 231027
#
# transform data in dir/ to json format

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

    dir = "data"
    files = os.listdir(dir)
    files_out = [file for file in files if "_out.txt" in file]
    files_tfidf = [file for file in files if "_tfidf.txt" in file]
    
    result = {}
    blacklist = []

    for input_file in files_out:
        with open(dir+"/"+input_file, 'r') as file:
            for line in file:
                line = eval(line)
                length = len(line)
                if length < 2:
                    blacklist.append(line[0])
                    continue
                newKey = False
                if line[0] not in result:
                    elements = line[1].split("-")
                    result[line[0]] = {elements[0]:{elements[1]:{"lines": [int(elements[2])]}}}  
                    newKey = True              
                for i in range(1, length):
                    if i == 1 and newKey == True:
                        continue
                    elements = line[i].split("-")
                    if elements[0] not in result[line[0]]:
                        result[line[0]][elements[0]] = {elements[1]:{"lines": [int(elements[2])]}}
                    elif elements[1] not in result[line[0]][elements[0]]:
                        result[line[0]][elements[0]][elements[1]] = {"lines": [int(elements[2])]}
                    elif int(elements[2]) not in result[line[0]][elements[0]][elements[1]]["lines"]:
                        result[line[0]][elements[0]][elements[1]]["lines"].append(int(elements[2]))

    for input_file in files_tfidf:
        with open(dir+"/"+input_file, 'r') as file: 
            ref = []
            for line in file:
                line = eval(line)
                length = len(line)
                if line[0] in blacklist:
                    continue
                elif line[0] not in result:
                    ref = line
                else:
                    for i in range(1, length):
                        if float(line[i]) > 0:
                            element0 = ref[i].split("-")[0]
                            element1 = ref[i].split("-")[1].split(".")[0]
                            # print(line[0], " ", element0, " ", element1)
                            if element0 in result[line[0]] and element1 in result[line[0]][element0] and "tfidf" not in result[line[0]][element0][element1]:
                                result[line[0]][element0][element1]["tfidf"] = float(line[i])
                            
    with open(dir+"/"+'data.json', 'w') as json_file:
        json.dump(result, json_file, indent=4, ensure_ascii=False)

    print("saved in " + dir + "/data.json")


if __name__ == "__main__":
   main(sys.argv[1:])  
