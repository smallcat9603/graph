import Pyro5
import Pyro5.client
import Pyro5.api
import time

# not require ns
uri = "PYRO:server@127.0.0.1:9091"
start = time.time()
obj = Pyro5.client.Proxy(uri)
batch = Pyro5.api.BatchProxy(obj)
for i in range(1000):
    try:
        batch.run()
    except Exception:
        print("Pyro traceback:")
        print("".join(Pyro5.errors.get_pyro_traceback()))
batch()
end = time.time()
print(f"time = {end-start}")