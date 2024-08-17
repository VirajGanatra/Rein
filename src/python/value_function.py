class ValueFunction:
    def predict(self, state):
        raise NotImplementedError

    def update(self, state, value):
        raise NotImplementedError