import Pyro4
import rw
import sys

nservers = 3 # server id = 0,1,2

if len(sys.argv) != 2 or not sys.argv[1].isdigit() or int(sys.argv[1]) > nservers-1 or int(sys.argv[1]) < 0:
    print("input server id [0-{0}]".format(nservers-1))
    sys.exit(1)

this = sys.argv[1]
next = str(int(sys.argv[1])%nservers + 1)
servername = "Server" + this
daemon = Pyro4.Daemon()
obj = rw.Walker(this, next)
uri = daemon.register(obj)
ns = Pyro4.locateNS()
ns.register(servername, uri)
# enter the service loop.
print("Server%s started ..." % this)
daemon.requestLoop()