import Pyro4

class Server(object):
  @Pyro4.expose
  def welcomeMessage(self, name):
    return ("Hi welcome " + str(name))

def startServer():
  server = Server()
  # daemon = Pyro4.Daemon()
  daemon = Pyro4.Daemon(host="0.0.0.0") # default port is random
  # ns = Pyro4.locateNS()
  ns = Pyro4.locateNS(host="127.0.0.1") # ns default port = 9090
  uri = daemon.register(server)
  ns.register("server", uri)
  print("Ready. Object uri =", uri)
  daemon.requestLoop()

if __name__ == "__main__":
  startServer()