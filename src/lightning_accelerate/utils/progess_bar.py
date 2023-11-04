import tqdm


class BaseProgessBar:
    def log(self, metrics: dict):
        raise NotImplementedError

    def update(self, n: int):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def get_logged_params(self) -> dict:
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class TQDMProgessBar(BaseProgessBar):
    def __init__(self, total: int, **kwargs):
        self.pbar = tqdm.tqdm(total=total, **kwargs)
        self.__log_params = dict()

    def log(self, metrics: dict):
        self.__log_params.update(metrics)
        self.pbar.set_postfix(self.__log_params)

    def update(self, n: int):
        self.pbar.update(n)

    def close(self):
        self.pbar.close()

    @property
    def tqdm(self):
        return self.pbar

    def get_logged_params(self):
        return self.__log_params.copy()
