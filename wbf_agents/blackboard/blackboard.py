class Blackboard:
    def __init__(self):
        self.data = {}
        self.locks = {}

    def post(self, key, value):
        self.data[key] = value

    def read_all(self):
        return self.data

    def lock_resource(self, key):
        self.locks[key] = True

    def unlock_resource(self, key):
        self.locks[key] = False

    def is_locked(self, key):
        return self.locks.get(key, False)
