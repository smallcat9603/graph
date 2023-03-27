import time


class ETA:
    def __init__(self):
        self.start = time.time()

    def eta(self, frac_done: float):
        """_summary_

        Args:
            frac_done float: % tasks already done in [0, 1]
        """
        return round(self.total_time() / frac_done * (1-frac_done), 3)

    def total_time(self):
        return time.time() - self.start
