import Pyro4
import sys

nhops = 100 # path length for each walker
if len(sys.argv) > 1 and int(sys.argv[1]) > 0:
    nhops = int(sys.argv[1])

obj = Pyro4.Proxy("PYRONAME:Server0")
try:
    obj.walk(["go"], nhops)
    print("Client starts ...")
except Exception:
    print("Pyro traceback:")
    print("".join(Pyro4.util.getPyroTraceback()))