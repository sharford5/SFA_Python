
class FeatureNode():
    def __init__(self, index, value):
        self.index = index
        self.value = value

    def getIndex(self):
        return self.index

    def getValue(self):
        return self.value

    def setValue(self, value):
        self.value = value

