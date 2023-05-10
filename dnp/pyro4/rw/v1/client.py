import Pyro5
import Pyro5.client
import sys

nhops = 100 # path length for each walker
if len(sys.argv) > 1 and int(sys.argv[1]) > 0:
    nhops = int(sys.argv[1])

# obj = Pyro5.client.Proxy("PYRONAME:Server0") # automatically look for ns first
obj = Pyro5.client.Proxy("PYRO:walker@127.0.0.1:9091") # connect to server directly (not need ns anymore)
try:
    print("Client starts ...")
    obj.walk(["go"], nhops)
    print("Client finished.")
except Exception:
    print("Pyro traceback:")
    print("".join(Pyro5.errors.get_pyro_traceback()))