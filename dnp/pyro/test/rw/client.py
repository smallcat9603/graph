# from __future__ import print_function
import Pyro4
obj = Pyro4.Proxy("PYRONAME:Server1")
print("Result = %s" % obj.process(["hello"]))