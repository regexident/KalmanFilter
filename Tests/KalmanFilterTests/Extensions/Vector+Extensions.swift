import Surge

@testable import KalmanFilter

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
}
