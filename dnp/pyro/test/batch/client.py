import Pyro5
import Pyro5.client
import Pyro5.api
import time
import platform

# not require ns
host = "127.0.0.1"
uname = platform.uname()
system = uname[0]
node = uname[1]
if system == "Linux":
    if "calc" in node:
        host = "10.52.10.9"
uri = "PYRO:server@" + host + ":9091"
start = time.time()
obj = Pyro5.client.Proxy(uri)
# batch = Pyro5.api.BatchProxy(obj)
# for i in range(1000):
try:
    # batch.run()
    obj.start(1000)
except Exception:
    print("Pyro traceback:")
    print("".join(Pyro5.errors.get_pyro_traceback()))
# batch()
end = time.time()
print(f"time = {end-start}")