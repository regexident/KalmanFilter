import Foundation

public protocol MotionModel {
    /// Calculate predicted state estimate
    ///
    /// ```
    /// x'(k) = A * x(k-1) + B * u(k)
    /// ```
    ///
    /// or more generally
    ///
    /// ```
    /// x'(k) = f(x(k-1))
    /// ```
    func apply(state x: Vector<Double>, input u: Vector<Double>) -> Vector<Double>
    
    /// Calculate jacobian matrix:
    ///
    /// ```
    /// F(k) = df(k)|
    ///        -----|
    ///         d(x)|
    ///             |x=X
    /// ```
    func jacobian(state x: Vector<Double>, input u: Vector<Double>) -> Matrix<Double>
}

public class LinearMotionModel {
    public let state: Matrix<Double>
    public let input: Matrix<Double>
    
    public init(
        state: Matrix<Double>,
        input: Matrix<Double>
    ) {
        self.state = state
        self.input = input
    }
}

extension LinearMotionModel: MotionModel {
    public func apply(state x: Vector<Double>, input u: Vector<Double>) -> Vector<Double> {
        let a = self.state
        let b = self.input
        return (a * x) + (b * u)
    }
    
    public func jacobian(state x: Vector<Double>, input u: Vector<Double>) -> Matrix<Double> {
        return self.state
    }
}

public class NonlinearMotionModel {
    public let function: (Vector<Double>, Vector<Double>) -> Vector<Double>
    public let jacobian: (Vector<Double>, Vector<Double>) -> Matrix<Double>
    
    public convenience init(
        dimensions: Dimensions,
        function: @escaping (Vector<Double>, Vector<Double>) -> Vector<Double>
    ) {
        self.init(dimensions: dimensions, function: function) { state, input in
            let jacobian = Jacobian(shape: (rows: dimensions.state, columns: dimensions.state))
            return jacobian.numeric(state: state) { function($0, input) }
        }
    }
    
    public init(
        dimensions: Dimensions,
        function: @escaping (Vector<Double>, Vector<Double>) -> Vector<Double>,
        jacobian: @escaping (Vector<Double>, Vector<Double>) -> Matrix<Double>
    ) {
        self.function = function
        self.jacobian = jacobian
    }
}

extension NonlinearMotionModel: MotionModel {
    public func apply(state x: Vector<Double>, input u: Vector<Double>) -> Vector<Double> {
        return self.function(x, u)
    }
    
    public func jacobian(state x: Vector<Double>, input u: Vector<Double>) -> Matrix<Double> {
        return self.jacobian(x, u)
    }
}
