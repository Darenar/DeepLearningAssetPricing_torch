class EarlyStopping:
    def __init__(self, thr: float = 0., min_wait: int = 300, minimize: bool = True):
        self.thr = thr
        self.min_wait = min_wait
        self.count = 0
        self.best_val = None
        self.minimize = minimize

    def __call__(self, metric_value: float) -> bool:
        if self.best_val is None:
            self.best_val = metric_value
            return False
        if self.minimize:
            if metric_value < self.best_val - self.thr:
                self.best_val = metric_value
                self.count = 0
            else:
                self.count += 1

        else:
            if metric_value > self.best_val + self.thr:
                self.best_val = metric_value
                self.count = 0
            else:
                self.count += 1
        if self.count >= self.min_wait:
            return True
        return False



