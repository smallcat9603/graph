import Pyro4

# ### option 1
# uri = input("What is the Pyro uri of the greeting object? ").strip()
# server = Pyro4.Proxy(uri)

# ### option 2
# server = Pyro4.Proxy("PYRONAME:server")

### option 3
uri = Pyro4.locateNS().lookup("server") 
server = Pyro4.Proxy(uri)

name = input("What is your name? ").strip()
print(server.welcomeMessage(name))