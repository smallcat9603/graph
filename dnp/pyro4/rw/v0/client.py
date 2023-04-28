import Pyro4
obj = Pyro4.Proxy("PYRONAME:Server0")
print("Result = %s" % obj.walk(["go"]))