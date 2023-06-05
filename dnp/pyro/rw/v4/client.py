import Pyro5
import Pyro5.client
import Pyro5.api
import sys, getopt, os, platform
import pandas as pd
import time
import threading
import json

def printUsage():
    print('Usage: python3 {0} -w [nwalkers] -s [nsteps] <number_of_servers>'.format(os.path.basename(__file__)))

start_times = []
stop_times = []
go_outs = []
merged_paths = [] # use list instead of dict to avoid "RuntimeError: dictionary changed size during iteration" error
finished = []

def count_finished(timestep, hosts, nhosts, Nwalkers):
    global start_times
    global stop_times
    global go_outs
    global merged_paths
    global finished
    while True:
        start_times = [] 
        stop_times = [] 
        go_outs = []
        merged_paths = []
        for host in range(nhosts):
            ip = hosts[host]
            uri = "PYRO:walker@" + ip
            obj = Pyro5.client.Proxy(uri) # connect to server directly (not need ns anymore)
            try:
                starttime, stoptime, goout, paths = obj.get_results()
                start_times.append(starttime)
                stop_times.append(stoptime)
                go_outs.append(goout)
                merged_paths += paths
            except Exception:
                print(f"Pyro traceback:\n{''.join(Pyro5.errors.get_pyro_traceback())}")  
        finished.append(len(merged_paths))
        print(finished[-1], end="\t", flush=True)
        if finished[-1] == Nwalkers or (len(finished) > 1 and finished[-1] == finished[-2]):
            break
        time.sleep(timestep)

def start_server(uri, nhops, id_start, id_end):
    # This will run in a thread. Create a proxy just for this thread:
    with Pyro5.client.Proxy(uri) as p:
        try:
            p.start_walkers(nhops, id_start, id_end)
        except Exception:
            print(f"Pyro traceback:\n{''.join(Pyro5.errors.get_pyro_traceback())}")           

def main(argv):
    nwalkers = 1 # number of walkers starting from each server
    nhops = 80 # path length for each walker

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

    # start servers
    for host in range(nhosts):
        id_start = host * nwalkers
        id_end = id_start + nwalkers
        ip = hosts[host]
        uri = "PYRO:walker@" + ip
        print(f"Client starts {nwalkers} Walkers[{id_start}-{id_end-1}] at Server{host} ({ip}) ...")
        t = threading.Thread(target=start_server, args=[uri, nhops, id_start, id_end])
        t.daemon = True
        t.start()
          
    # total number of walkers = nhosts * nwalkers
    Nwalkers = nhosts*nwalkers
    t_cf = threading.Thread(target=count_finished, args=[1, hosts, nhosts, Nwalkers])
    t_cf.daemon = True
    t_cf.start() 
    t_cf.join()
    print("Done.")

    # output results
    runtime = max(stop_times)-min(start_times)
    print(f"time = {runtime}")
    print(f"goout = {sum(go_outs)}")
    timestamp = int(min(start_times))
    dir = "../log"
    if not os.path.exists(dir):
        os.makedirs(dir)
    jsonfile = dir + f"/t{timestamp}_n{nhosts}.json"
    txtfile = dir + f"/t{timestamp}_n{nhosts}.txt"
    with open(jsonfile, 'w') as file:
        json.dump(dict(merged_paths), file)
    print(f"paths saved in {jsonfile}")
    line = f"{runtime}\t{len(merged_paths)}\t{sum(go_outs)}"
    for each in go_outs:
        line += f"\t{each}"
    line += "\n"
    with open(txtfile, 'w') as file:
        file.write(line)
        for item in finished:
            file.write(str(item) + '\n')
    print(f"statistics saved in {txtfile}")

if __name__ == "__main__":
    main(sys.argv[1:])