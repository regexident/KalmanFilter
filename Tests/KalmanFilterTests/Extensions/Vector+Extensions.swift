import Surge

extension Vector where Scalar == Float {
    internal init(random dimensions: Int) {
        let scalars = (0..<dimensions).map { _ in Float.random(in: 0.0..<1.0) }
        self.init(scalars)
    }

    internal init(random dimensions: Int, in range: Range<Float>) {
        let scalars = (0..<dimensions).map { _ in Float.random(in: range) }
        self.init(scalars)
    }

    internal init(gaussianRandom dimensions: Int) {
        let scalars = (0..<dimensions).map { _ in Float.gaussianRandom() }
        self.init(scalars)
    }

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
    internal init(random dimensions: Int) {
        let scalars = (0..<dimensions).map { _ in Double.random(in: 0.0..<1.0) }
        self.init(scalars)
    }
    
    internal init(random dimensions: Int, in range: Range<Double>) {
        let scalars = (0..<dimensions).map { _ in Double.random(in: range) }
        self.init(scalars)
    }
    
    internal init(gaussianRandom dimensions: Int) {
        let scalars = (0..<dimensions).map { _ in Double.gaussianRandom() }
        self.init(scalars)
    }

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
