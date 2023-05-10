import Pyro5
import Pyro5.client
import sys
import pandas as pd

# read hosts from file
columns = ["server_id", "ip_port"]
hostfile = pd.read_csv("hosts.txt", comment="#", sep="\s+", names=columns)
nhosts = len(hostfile) # server id = 0,1,2
hosts = {}
for row in range(nhosts):
    hosts[int(hostfile["server_id"][row])] = hostfile["ip_port"][row]

nhops = 100 # path length for each walker
if len(sys.argv) > 1 and int(sys.argv[1]) > 0:
    nhops = int(sys.argv[1])

for walker in range(10):
    # obj = Pyro5.client.Proxy("PYRONAME:Server0") # automatically look for ns first
    uri = "PYRO:walker@" + hosts[0]
    obj = Pyro5.client.Proxy(uri) # connect to server directly (not need ns anymore)
    try:
        print("Client{0} starts ...".format(walker))
        obj.walk(["go"], nhops, walker)
        print("Client{0} finished.".format(walker))
    except Exception:
        print("Pyro traceback:")
        print("".join(Pyro5.errors.get_pyro_traceback()))