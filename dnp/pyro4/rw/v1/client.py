import Pyro5
import Pyro5.client
import sys, getopt, os
import pandas as pd

def printUsage():
    print('Usage: python3 {0} -w [nwalkers] -s [nsteps]'.format(os.path.basename(__file__)))

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

    # read hosts from file
    columns = ["server_id", "ip_port"]
    hostfile = pd.read_csv("hosts.txt", comment="#", sep="\s+", names=columns)
    nhosts = len(hostfile) # server id = 0,1,2, ...
    hosts = {}
    for row in range(nhosts):
        hosts[int(hostfile["server_id"][row])] = hostfile["ip_port"][row]

    # total number of walkers = nhosts * nwalkers
    for host in range(nhosts):
        id_start = host * nwalkers
        id_end = id_start + nwalkers
        for walker in range(id_start, id_end):
            # obj = Pyro5.client.Proxy("PYRONAME:Server0") # automatically look for ns first
            ip = hosts[host]
            uri = "PYRO:walker@" + ip
            obj = Pyro5.client.Proxy(uri) # connect to server directly (not need ns anymore)
            try:
                print("Client starts Walker{0} at Server{1} ({2}) ...".format(walker, host, ip))
                obj.walk(["go"], nhops, walker)
                # print("Client{0} finished.".format(walker))
            except Exception:
                print("Pyro traceback:")
                print("".join(Pyro5.errors.get_pyro_traceback()))

if __name__ == "__main__":
    main(sys.argv[1:])