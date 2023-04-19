import Pyro4

# ### option 1
# uri = input("What is the Pyro uri of the greeting object? ").strip()

# ### option 2
# uri = "PYRONAME:server"

# ### option 3
# uri = Pyro4.locateNS().lookup("server") 

# ### option 4
# uri = Pyro4.locateNS(host="127.0.0.1").lookup("server") # ns default port = 9090

### option 5
uri = "PYRONAME:server@127.0.0.1" # "PYRONAME:server@127.0.0.1:9090"


server = Pyro4.Proxy(uri)
name = input("What is your name? ").strip()
print(server.welcomeMessage(name))