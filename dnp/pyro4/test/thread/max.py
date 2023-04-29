import threading
import time

def mythread():
  time.sleep(1000)

def main():
  threads = 0
  y = 1000000
  start = time.time()
  for i in range(y):
    try:
      x = threading.Thread(target=mythread, daemon=True)
      threads += 1
      x.start()
    except RuntimeError:
      break
  print("{0} threads created.".format(threads))
  end = time.time()
  print("time = {0}".format(end-start))

if __name__ == "__main__":
  main()