import Pyro4

# ### option 1 (require server ip)
# uri = input("What is the Pyro uri of the greeting object? ").strip() # e.g., "PYRO:obj_7f2e746b12934b55a119b9de946cf6e7@0.0.0.0:56739"

# ### option 2 (name server is local)
# uri = "PYRONAME:server"

# ### option 3 (name server is local)
# uri = Pyro4.locateNS().lookup("server") 

# ### option 4 (require name server ip)
# uri = Pyro4.locateNS(host="127.0.0.1").lookup("server") # ns default port = 9090

### option 5 (require name server ip)
uri = "PYRONAME:server@127.0.0.1" # "PYRONAME:server@127.0.0.1:9090"


server = Pyro4.Proxy(uri)
name = input("What is your name? ").strip()
print(server.welcomeMessage(name))