import Pyro4
import chainTopology
import sys

nservers = 3 # server id = 1,2,3

if len(sys.argv) != 2 or not sys.argv[1].isdigit() or int(sys.argv[1]) > nservers or int(sys.argv[1]) < 1:
    print("input server id [1-{0}]".format(nservers))
    sys.exit(1)

this = sys.argv[1]
next = str(int(sys.argv[1])%nservers + 1)
servername = "Server" + this
daemon = Pyro4.Daemon()
obj = chainTopology.Chain(this, next)
uri = daemon.register(obj)
ns = Pyro4.locateNS()
ns.register(servername, uri)
# enter the service loop.
print("Server%s started ..." % this)
daemon.requestLoop()