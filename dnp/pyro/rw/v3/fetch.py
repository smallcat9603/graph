import Pyro5
import Pyro5.client
import sys, getopt, os, platform
import pandas as pd
import time
import json

def printUsage():
    print('Usage: python3 {0} <number_of_servers>'.format(os.path.basename(__file__)))

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h") 
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
    nhosts = 3
    if len(args) == 1 and 1 <= int(args[0]) <= 8:
        nhosts = int(args[0])   
    else:        
        print("input number of servers (<= 8)")
        sys.exit(1)      

    # match hostfile
    filename = ""
    uname = platform.uname()
    system = uname[0]
    node = uname[1]
    if system == "Darwin":
        filename = "hosts_local.txt"
    elif system == "Linux":
        if "calc" in node:
            filename = "hosts_calc.txt"

    # read hosts from file
    columns = ["server_id", "ip_port"]
    hostfile = pd.read_csv(filename, comment="#", sep="\s+", names=columns)
    hosts = {}
    for row in range(nhosts):
        hosts[int(hostfile["server_id"][row])] = hostfile["ip_port"][row]

    # fetch results from each server
    start_times = []
    stop_times = []
    merged_paths = {}
    for host in range(nhosts):
        ip = hosts[host]
        uri = "PYRO:walker@" + ip
        obj = Pyro5.client.Proxy(uri) # connect to server directly (not need ns anymore)
        try:
            starttime, stoptime, paths = obj.get_results()
            start_times.append(starttime)
            stop_times.append(stoptime)
            merged_paths.update(paths)
        except Exception:
            print("Pyro traceback:")
            print("".join(Pyro5.errors.get_pyro_traceback()))

    # output resuts
    if len(set(start_times)) == 1:
        print("timestamp synchronized")
    else:
        print("timestamp not synchronized")
    print(f"time = {max(stop_times)-max(start_times)}")
    timestamp = int(max(start_times))
    dir = "../log"
    if not os.path.exists(dir):
        os.makedirs(dir)
    filepath = dir + f"/t{timestamp}_n{nhosts}.log"
    with open(filepath, 'w') as file:
        json.dump(merged_paths, file)
    print(f"paths saved in {filepath}")
    
if __name__ == "__main__":
    main(sys.argv[1:])