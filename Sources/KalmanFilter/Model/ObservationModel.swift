import Foundation

import Surge

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
    
    /// Validate the model for a given dimensional environment
    ///
    /// - Parameters:
    ///   - dimensions: the environment's dimensions
    func validate(for dimensions: Dimensions) throws
}

public struct LinearObservationModel {
    public enum Error: Swift.Error {
        case state(MatrixError)
    }
    
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
    
    public func validate(for dimensions: Dimensions) throws {
        guard self.state.columns == dimensions.state else {
            let actual = self.state.columns
            let expected = dimensions.state
            throw Error.state(.invalidColumnCount(
                message: "Expected \(expected) columns in `self.state`, found \(actual)"
            ))
        }
        
        guard self.state.rows == dimensions.observation else {
            let actual = self.state.rows
            let expected = dimensions.observation
            throw Error.state(.invalidRowCount(
                message: "Expected \(expected) columns in `self.state`, found \(actual)"
            ))
        }
    }
}

public class NonlinearObservationModel {
    public enum Error: Swift.Error {
        case invalid(message: String)
    }
    
    public let function: (Vector<Double>) -> Vector<Double>
    public let jacobian: (Vector<Double>) -> Matrix<Double>
    
    public convenience init(dimensions: Dimensions, function: @escaping (Vector<Double>) -> Vector<Double>) {
        self.init(function: function) { state in
            let jacobian = Jacobian(rows: dimensions.observation, columns: dimensions.state)
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
    
    public func validate(for dimensions: Dimensions) throws {
        #if DEBUG
        let state = Vector(dimensions: dimensions.state, repeatedValue: 0.0)
        
        let observation = self.apply(state: state)
        
        if observation.dimensions != dimensions.observation {
            let actual = observation.dimensions
            let expected = dimensions.observation
            throw Error.invalid(
                message: "Expected output vector of \(expected) dimensions, found \(actual)"
            )
        }
        #endif
    }
}
