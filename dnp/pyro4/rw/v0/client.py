import Pyro4
import Pyro4.util

obj = Pyro4.Proxy("PYRONAME:Server0")
try:
    obj.walk(["go"])
    print("Client starts ...")
except Exception:
    print("Pyro traceback:")
    print("".join(Pyro4.util.getPyroTraceback()))