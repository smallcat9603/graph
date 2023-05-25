import time
import threading

def loop():
    for j in range(10000):
        print(str(j))

start = time.time()
for i in range(1000):
    loop()
    # t = threading.Timer(0, loop)
    # t.start()
end = time.time()
print(f"time = {end-start}")