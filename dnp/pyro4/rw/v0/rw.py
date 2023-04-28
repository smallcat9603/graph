# from __future__ import print_function
import Pyro4
import random

def generate_random_numbers():
    num_of_numbers = random.randint(1, 4)
    random_numbers = []
    for i in range(num_of_numbers):
        random_numbers.append(random.randint(1, 100))
    return random_numbers

class Walker(object):
    def __init__(self, name, next):
        self.name = name
        self.nextName = next
        self.next = None
    @Pyro4.expose
    def process(self, message):
        if self.next is None:
            self.next = Pyro4.Proxy("PYRONAME:Server" + self.nextName)
        if self.name in message:
            rn = generate_random_numbers()
            print("Server{0} walks through nodes: {1}".format(self.name, rn))
            print("Walker stopped at server%s" % self.name)
            return rn
        else:
            message.append(self.name)
            result = self.next.process(message)
            rn = generate_random_numbers()
            print("Server{0} walks through nodes: {1}".format(self.name, rn))
            result = rn + result
            print("Walker walks from server%s to server%s" % (self.name, self.nextName))
            return result
