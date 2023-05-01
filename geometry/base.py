from abc import ABC, abstractmethod

class Geometry:
    @abstractmethod
    def dim(self):
        """All geometry objects must be embeddable in some dimension."""

    @abstractmethod
    def __hash__(self):
        """All geometry objects must be hashable."""
