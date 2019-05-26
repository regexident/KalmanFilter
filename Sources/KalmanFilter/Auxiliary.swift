import Foundation

public struct Dimensions {
    public let state: Int
    public let input: Int
    public let output: Int
    
    public init(state: Int, input: Int, output: Int) {
        assert(state >= 1)
        assert(input >= 1)
        assert(output >= 1)
        
        self.state = state
        self.input = input
        self.output = output
    }
    
    public init(uniform: Int) {
        assert(uniform >= 1)
        
        self.init(state: uniform, input: uniform, output: uniform)
    }
}

extension Dimensions: CustomStringConvertible {
    public var description: String {
        return "{ state: \(self.state), input: \(self.input), output: \(self.output) }"
    }
}

//public enum Model {
//    case linear(ModelProtocol)
//    case nonlinear(NonLinearModelProtocol)
//}
//
//public protocol ModelProtocol {
//    func apply(_ x: Vector<Double>) -> Vector<Double>
//}
//
//public protocol NonLinearModelProtocol: ModelProtocol {
//    func jacobian(state x: Vector<Double>, rows r: Int) -> Matrix<Double>
//}
//
//extension Matrix: ModelProtocol where Scalar == Double {
//    public func apply(_ x: Vector<Scalar>) -> Vector<Scalar> {
//        return self * x
//    }
//}
//
//extension Matrix: NonLinearModelProtocol where Scalar == Double {}
//
//extension NonLinearModelProtocol {
//    public func jacobian(state x: Vector<Double>, rows r: Int) -> Matrix<Double> {
//        return self.numericJacobian(state: x, rows: r)
//    }
//
//    public func numericJacobian(
//        state x: Vector<Double>,
//        rows r: Int,
//        delta t: Double = 0.000001
//    ) -> Matrix<Double> {
//        var jacobian: Matrix<Double> = .init(rows: r, columns: x.rows)
//        var dx: Vector<Double> = .init(rows: x.rows)
//        for i in 0..<x.rows {
//            dx[i] = t
//            let column = (self.apply(x + dx) - self.apply(x - dx)) / (t * 2.0)
//            jacobian[i] = column
//            dx[i] = 0.0
//        }
//        return jacobian
//    }
//}
//
//public struct NonLinearModel: NonLinearModelProtocol {
//    let mapping: (Vector<Double>) -> Vector<Double>
//
//    public init(_ mapping: @escaping (Vector<Double>) -> Vector<Double>) {
//        self.mapping = mapping
//    }
//
//    public func apply(_ x: Vector<Double>) -> Vector<Double> {
//        return self.mapping(x)
//    }
//}


//public protocol ObservationModel {
//    func apply(state x: Vector<Double>) -> (x: Vector<Double>, a: Matrix<Double>)
//}
