import Foundation

public protocol ObservationModel {
    func apply(state x: Vector<Double>) -> Vector<Double>
    func apply(state x: Vector<Double>) -> (z: Vector<Double>, h: Matrix<Double>)
}

extension ObservationModel {
    public func apply(state x: Vector<Double>) -> Vector<Double> {
        return self.apply(state: x).z
    }
}

public protocol MatrixObservationModel: ObservationModel {
    func h(state x: Vector<Double>) -> Matrix<Double>
}

public struct StaticMatrixObservationModel {
    /// Output matrix (aka `H`, or sometimes `C`)
    ///
    /// This matrix influences the Kalman Gain.
    ///
    /// Note: It maps from a state to an output.
    ///
    /// Default: identity matrix.
    let h: Matrix<Double>
    
    public init(h: Matrix<Double>) {
        self.h = h
    }
}

extension StaticMatrixObservationModel: MatrixObservationModel {
    public func h(state x: Vector<Double>) -> Matrix<Double> {
        return self.h
    }
}

extension StaticMatrixObservationModel: CustomStringConvertible {
    public var description: String {
        return [
            "h:\n\(self.h)",
        ].joined(separator: "\n")
    }
}

public struct DynamicMatrixObservationModel {
    /// Output matrix (aka `H`, or sometimes `C`)
    ///
    /// This matrix influences the Kalman Gain.
    ///
    /// Note: It maps from a state to an output.
    ///
    /// Default: identity matrix.
    let h: (Vector<Double>) -> Matrix<Double>
    
    public init(
        h: @escaping (Vector<Double>) -> Matrix<Double>
    ) {
        self.h = h
    }
}

extension DynamicMatrixObservationModel: MatrixObservationModel {
    public func h(state x: Vector<Double>) -> Matrix<Double> {
        return self.h(x)
    }
}

extension DynamicMatrixObservationModel: CustomStringConvertible {
    public var description: String {
        return [
            "a: dynamic",
            "b: dynamic",
        ].joined(separator: "\n")
    }
}

extension MatrixObservationModel {
    public func apply(state x: Vector<Double>) -> (z: Vector<Double>, h: Matrix<Double>) {
        let h = self.h(state: x)
        
        // Calculate z prediction:
        // z'(k) = H * x'(k)
        let z = h * x
        
        return (z: z, h: h)
    }
}

public struct FunctionObservationModel: ObservationModel {
    let output: Int
    let h: (Vector<Double>) -> Vector<Double>
    
    public init(output: Int, _ h: @escaping (Vector<Double>) -> Vector<Double>) {
        self.output = output
        self.h = h
    }
    
    public func apply(state x: Vector<Double>) -> (z: Vector<Double>, h: Matrix<Double>) {
        let jacobian = Jacobian(shape: (columns: x.rows, rows: self.output))
        
        // Calculate jacobian matrix:
        // F(k) = df(k)|
        //        -----|
        //         d(x)|
        //             |x=X
        let h = jacobian.numeric(state: x) { self.h($0) }
        
        // Calculate z prediction:
        // z'(k) = H * x'(k)
        let z = self.h(x)
        
        return (z: z, h: h)
    }
}

extension FunctionObservationModel: CustomStringConvertible {
    public var description: String {
        return [
            "h: non-linear",
        ].joined(separator: "\n")
    }
}
