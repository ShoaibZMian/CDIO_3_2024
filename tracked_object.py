class TrackedObject:
    def __init__(self, object_type, contour, centroid):
        self.object_type = object_type
        self.contour = contour
        self.centroid = centroid
        self.last_seen = 0  # Frames since last seen