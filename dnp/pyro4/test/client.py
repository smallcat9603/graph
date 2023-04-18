import Pyro4

# ### option 1
# uri = input("What is the Pyro uri of the greeting object? ").strip()
# server = Pyro4.Proxy(uri)

# ### option 2
# server = Pyro4.Proxy("PYRONAME:server")

# ### option 3
# uri = Pyro4.locateNS().lookup("server") 
# server = Pyro4.Proxy(uri)

### option 4
# uri = Pyro4.locateNS(host="127.0.0.1").lookup("server") 
uri = Pyro4.locateNS(host="192.168.3.39").lookup("server") 
server = Pyro4.Proxy(uri)

name = input("What is your name? ").strip()
print(server.welcomeMessage(name))