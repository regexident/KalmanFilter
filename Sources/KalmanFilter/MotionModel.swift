import Foundation

public protocol MotionModel {
    func apply(state x: Vector<Double>, input u: Vector<Double>) -> Vector<Double>
    func apply(state x: Vector<Double>, input u: Vector<Double>) -> (x: Vector<Double>, a: Matrix<Double>)
}

extension MotionModel {
    public func apply(state x: Vector<Double>, input u: Vector<Double>) -> Vector<Double> {
        return self.apply(state: x, input: u).x
    }
}

public protocol MatrixMotionModel: MotionModel {
    func a(state x: Vector<Double>) -> Matrix<Double>
    func b(state x: Vector<Double>, input u: Vector<Double>) -> Matrix<Double>
}

public struct StaticMatrixMotionModel {
    /// Transition matrix (aka `A`, or sometimes `F` or `Φ`)
    ///
    /// This matrix influences the output vector.
    ///
    /// Note: It maps from a state to a state.
    ///
    /// Default: identity matrix.
    let a: Matrix<Double>
    
    /// Control matrix (aka `B`, or sometimes `G`)
    ///
    /// This matrix influences the input vector.
    ///
    /// Note: It maps from an input to a state.
    ///
    /// Default: identity matrix.
    let b: Matrix<Double>
    
    public init(a: Matrix<Double>, b: Matrix<Double>) {
        assert(a.rows == a.columns, "Expected matrix `a` to be square.")
        assert(b.rows == a.columns, "Expected matrix `b` to have as many rows as matrix `a` has columns.")
        
        self.a = a
        self.b = b
    }
}

extension StaticMatrixMotionModel: MatrixMotionModel {
    public func a(state x: Vector<Double>) -> Matrix<Double> {
        return self.a
    }
    
    public func b(state x: Vector<Double>, input u: Vector<Double>) -> Matrix<Double> {
        return self.b
    }
}

extension StaticMatrixMotionModel: CustomStringConvertible {
    public var description: String {
        return [
            "a:\n\(self.a)",
            "b:\n\(self.b)",
        ].joined(separator: "\n")
    }
}

public struct DynamicMatrixMotionModel {
    /// Transition matrix (aka `A`, or sometimes `F` or `Φ`)
    ///
    /// This matrix influences the output vector.
    ///
    /// Note: It maps from a state to a state.
    ///
    /// Default: identity matrix.
    let a: (Vector<Double>) -> Matrix<Double>
    
    /// Control matrix (aka `B`, or sometimes `G`)
    ///
    /// This matrix influences the input vector.
    ///
    /// Note: It maps from an input to a state.
    ///
    /// Default: identity matrix.
    let b: (Vector<Double>, Vector<Double>) -> Matrix<Double>
    
    public init(
        a: @escaping (Vector<Double>) -> Matrix<Double>,
        b: @escaping (Vector<Double>, Vector<Double>) -> Matrix<Double>
    ) {
        self.a = a
        self.b = b
    }
}

extension DynamicMatrixMotionModel: MatrixMotionModel {
    public func a(state x: Vector<Double>) -> Matrix<Double> {
        return self.a(x)
    }
    
    public func b(state x: Vector<Double>, input u: Vector<Double>) -> Matrix<Double> {
        return self.b(x, u)
    }
}

extension DynamicMatrixMotionModel: CustomStringConvertible {
    public var description: String {
        return [
            "a: dynamic",
            "b: dynamic",
        ].joined(separator: "\n")
    }
}

extension MatrixMotionModel {
    public func apply(state x: Vector<Double>, input u: Vector<Double>) -> (x: Vector<Double>, a: Matrix<Double>) {
        let a = self.a(state: x)
        let b = self.b(state: x, input: u)
        
        // Calculate predicted state estimate
        // x'(k) = A * x(k-1) + B * u(k)
        let x = (a * x) + (b * u)
        
        return (x: x, a: a)
    }
}

public struct FunctionMotionModel {
    let f: (Vector<Double>, Vector<Double>) -> Vector<Double>
    let j: (Vector<Double>, Vector<Double>) -> Matrix<Double>
    
    public init(
        f: @escaping (Vector<Double>, Vector<Double>) -> Vector<Double>
    ) {
        self.init(f: f, j: { state, input in
            let jacobian = Jacobian(shape: (columns: state.rows, rows: state.rows))
            return jacobian.numeric(state: state) { f($0, input) }
        })
    }
    
    public init(
        f: @escaping (Vector<Double>, Vector<Double>) -> Vector<Double>,
        j: @escaping (Vector<Double>, Vector<Double>) -> Matrix<Double>
    ) {
        self.f = f
        self.j = j
    }
}

extension FunctionMotionModel: MotionModel {
    public func apply(state x: Vector<Double>, input u: Vector<Double>) -> (x: Vector<Double>, a: Matrix<Double>) {
        // Calculate jacobian matrix:
        // F(k) = df(k)|
        //        -----|
        //         d(x)|
        //             |x=X
        let a = self.j(x, u)
        
        // Calculate predicted state estimate
        // x'(k) = f(x(k-1))
        let x = self.f(x, u)
        
        return (x: x, a: a)
    }
}
