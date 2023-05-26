import Pyro5
import Pyro5.client
import Pyro5.api
import sys, getopt, os, platform
import pandas as pd
import time
import json

def printUsage():
    print('Usage: python3 {0} -w [nwalkers] -s [nsteps] <number_of_servers>'.format(os.path.basename(__file__)))

def main(argv):
    nwalkers = 1 # number of walkers starting from each server
    nhops = 100 # path length for each walker

    try:
        opts, args = getopt.getopt(argv, "hw:s:") 
    except getopt.GetoptError:
        printUsage()
        sys.exit(1)
    for opt, arg in opts:
        if opt == '-h':
            printUsage()
            sys.exit()
        elif opt == '-w':
            nwalkers = int(arg)
        elif opt == '-s':
            nhops = int(arg)
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

    # total number of walkers = nhosts * nwalkers
    start_time = time.time()
    for host in range(nhosts):
        id_start = host * nwalkers
        id_end = id_start + nwalkers
        ip = hosts[host]
        uri = "PYRO:walker@" + ip
        # obj = Pyro5.client.Proxy("PYRONAME:Server0") # automatically look for ns first
        obj = Pyro5.client.Proxy(uri) # connect to server directly (not need ns anymore)
        # batch = Pyro5.api.BatchProxy(obj)
        try:
            # for walker in range(id_start, id_end):
            #     # obj.walk(["go"], nhops, walker)
            #     batch.walk([f"go_{start_time}"], nhops, walker)
            #     # print("Client{0} finished.".format(walker))
            # batch()
            print(f"Client starts {nwalkers} Walkers[{id_start}-{id_end-1}] at Server{host} ({ip}) ...")
            obj.start(start_time, nhops, id_start, id_end)
        except Exception:
            print("Pyro traceback:")
            print("".join(Pyro5.errors.get_pyro_traceback()))

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
    dir = "../log"
    if not os.path.exists(dir):
        os.makedirs(dir)
    filepath = dir + f"/t{int(start_time)}_w{nwalkers}_s{nhops}_n{nhosts}.log"
    with open(filepath, 'w') as file:
        json.dump(merged_paths, file)
    print(f"paths saved in {filepath}")
    

if __name__ == "__main__":
    main(sys.argv[1:])