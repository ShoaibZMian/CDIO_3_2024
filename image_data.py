class ProcessedImageData():
    def __init__(self):
        self.is_fresh: bool = False
        self.id = 0
        self.tracked_objects: list[TrackedObject] = list[TrackedObject]()
        self.frame = None

class TrackedObject:
    def __init__(self, object_type, contour, centroid):
        self.object_type = object_type
        self.contour = contour
        self.centroid = centroid
        self.last_seen = 0  # Frames since last seen