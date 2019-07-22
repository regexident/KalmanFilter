import Foundation

public protocol ObservationModel {
    /// Calculate z prediction:
    ///
    /// ```
    /// z'(k) = H * x'(k)
    /// ```
    func apply(state x: Vector<Double>) -> Vector<Double>
    
    /// Calculate jacobian matrix:
    ///
    /// ```
    /// H(k) = dh(k)|
    ///        -----|
    ///         d(x)|
    ///             |x=X
    /// ```
    func jacobian(state x: Vector<Double>) -> Matrix<Double>
}

public struct LinearObservationModel {
    public let state: Matrix<Double>
    
    public init(state: Matrix<Double>) {
        self.state = state
    }
}

extension LinearObservationModel: ObservationModel {
    public func apply(state x: Vector<Double>) -> Vector<Double> {
        let h = self.state
        return h * x
    }
    
    public func jacobian(state x: Vector<Double>) -> Matrix<Double> {
        return self.state
    }
}

public class NonlinearObservationModel {
    public let function: (Vector<Double>) -> Vector<Double>
    public let jacobian: (Vector<Double>) -> Matrix<Double>
    
    public convenience init(dimensions: Dimensions, function: @escaping (Vector<Double>) -> Vector<Double>) {
        self.init(function: function) { state in
            let jacobian = Jacobian(shape: (rows: dimensions.observation, columns: dimensions.state))
            return jacobian.numeric(state: state) { function($0) }
        }
    }
    
    public init(
        function: @escaping (Vector<Double>) -> Vector<Double>,
        jacobian: @escaping (Vector<Double>) -> Matrix<Double>
    ) {
        self.function = function
        self.jacobian = jacobian
    }
}

extension NonlinearObservationModel: ObservationModel {
    public func apply(state x: Vector<Double>) -> Vector<Double> {
        return self.function(x)
    }
    
    public func jacobian(state x: Vector<Double>) -> Matrix<Double> {
        return self.jacobian(x)
    }
}
