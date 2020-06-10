import abc


class FeatureBasedLocationDetector:
    @abc.abstractmethod
    def map_features(self, map_img):
        pass

    @abc.abstractmethod
    def loc_features(self, loc_img):
        pass

    @abc.abstractmethod
    def detect_location(self, map_img, loc_img, map_features, loc_features):
        pass
