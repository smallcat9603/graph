import Pyro5
import Pyro5.server
import platform

class Server(object):
  @Pyro5.server.expose
  @Pyro5.server.oneway
  def welcomeMessage(self, name):
    return ("Hi welcome " + str(name))

  @Pyro5.server.expose
  def run(self):
    for i in range(10000):
      print(str(i))

  @Pyro5.server.expose
  def start(self, num):
    for i in range(num):
      self.run()

def startServer():
  host = "127.0.0.1"
  uname = platform.uname()
  system = uname[0]
  node = uname[1]
  if system == "Linux":
    if "calc" in node:
        host = "10.52.10.9"
  port = 9091
  daemon = Pyro5.server.Daemon(host=host, port=port)
  obj = Server()
  uri = daemon.register(obj, objectId="server") # default objectId is random like obj_79549b4c52dc43ffaaa486b76b25c2af
  print(f"Server started ...")
  daemon.requestLoop()

if __name__ == "__main__":
  startServer()