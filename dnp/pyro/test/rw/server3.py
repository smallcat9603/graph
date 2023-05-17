# from __future__ import print_function
import Pyro4
import chainTopology
this = "3"
next = "1"
servername = "Server" + this
daemon = Pyro4.Daemon()
obj = chainTopology.Chain(this, next)
uri = daemon.register(obj)
ns = Pyro4.locateNS()
ns.register(servername, uri)
# enter the service loop.
print("Server%s started ..." % this)
daemon.requestLoop()