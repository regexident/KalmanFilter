import Surge

extension Vector where Scalar == Float {
    public func magnitude() -> Scalar {
        return self.magnitudeSquared().squareRoot()
    }

    public func magnitudeSquared() -> Scalar {
        return Surge.dot(self, self)
    }

    public func distance(to other: Vector<Scalar>) -> Scalar {
        return Surge.dist(self, other)
    }

    public func distanceSquared(to other: Vector<Scalar>) -> Scalar {
        return Surge.distSq(self, other)
    }
}

extension Vector where Scalar == Double {
    public func magnitude() -> Scalar {
        return self.magnitudeSquared().squareRoot()
    }

    public func magnitudeSquared() -> Scalar {
        return Surge.dot(self, self)
    }

    public func distance(to other: Vector<Scalar>) -> Scalar {
        return Surge.dist(self, other)
    }

    public func distanceSquared(to other: Vector<Scalar>) -> Scalar {
        return Surge.distSq(self, other)
    }
}
