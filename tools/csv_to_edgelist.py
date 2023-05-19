# smallcat 230517

import igraph as ig
import pandas as pd
import sys, getopt, os

def printUsage():
    print('Usage: python3 {0} <edgelistfile>'.format(os.path.basename(__file__)))

def main(argv):
    index_col = None
    header = None
    try:
        opts, args = getopt.getopt(argv, "hIH") # opts = [("-h", " "), ("-I", " "), ("-H", " ")], args = [edgelistfile]
    except getopt.GetoptError:
        printUsage()
        sys.exit(1)
    for opt, arg in opts:
        if opt == '-h':
            printUsage()
            sys.exit()
        elif opt == '-I':
            index_col = 0    
        elif opt == '-H':
            header = 0
        else:
            printUsage()
            sys.exit(1)
    if len(args) != 1:
            printUsage()
            sys.exit(1)       
    edgefile = args[0]

    data = pd.read_csv(edgefile, index_col=index_col, header=header)
    txt = edgefile[:-len(".csv")] + ".txt" # use removesuffix in python3.9
    data.to_csv(txt, sep='\t', index=False, header=False)
    print(txt + " generated.")

if __name__ == "__main__":
   main(sys.argv[1:])  
