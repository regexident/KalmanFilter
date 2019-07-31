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
    func apply(state x: Vector<Double>, control u: Vector<Double>) -> Vector<Double>
    
    /// Calculate jacobian matrix:
    ///
    /// ```
    /// F(k) = df(k)|
    ///        -----|
    ///         d(x)|
    ///             |x=X
    /// ```
    func jacobian(state x: Vector<Double>, control u: Vector<Double>) -> Matrix<Double>
    
    /// Validate the model for a given dimensional environment
    ///
    /// - Parameters:
    ///   - dimensions: the environment's dimensions
    func validate(for dimensions: Dimensions) throws
}

public class LinearMotionModel {
    public enum Error: Swift.Error {
        case state(MatrixError)
        case control(MatrixError)
    }
    
    public let state: Matrix<Double>
    public let control: Matrix<Double>
    
    public init(
        state: Matrix<Double>,
        control: Matrix<Double>
    ) {
        self.state = state
        self.control = control
    }
}

extension LinearMotionModel: MotionModel {
    public func apply(state x: Vector<Double>, control u: Vector<Double>) -> Vector<Double> {
        let a = self.state
        let b = self.control
        return (a * x) + (b * u)
    }
    
    public func jacobian(state x: Vector<Double>, control u: Vector<Double>) -> Matrix<Double> {
        return self.state
    }
    
    public func validate(for dimensions: Dimensions) throws {
        guard self.state.columns == dimensions.state else {
            throw Error.state(.invalidColumnCount(
                message: "Expected \(dimensions.state) columns in `self.state`, found \(self.state.columns)"
            ))
        }
        
        guard self.state.rows == dimensions.state else {
            throw Error.state(.invalidRowCount(
                message: "Expected \(dimensions.state) columns in `self.state`, found \(self.state.rows)"
            ))
        }
        
        guard self.control.columns == dimensions.control else {
            throw Error.control(.invalidColumnCount(
                message: "Expected \(dimensions.control) columns in `self.control`, found \(self.control.columns)"
            ))
        }
        
        guard self.control.rows == dimensions.state else {
            throw Error.control(.invalidRowCount(
                message: "Expected \(dimensions.state) columns in `self.control`, found \(self.state.rows)"
            ))
        }
    }
}

public class NonlinearMotionModel {
    public enum Error: Swift.Error {
        case invalid(message: String)
    }
    
    public let function: (Vector<Double>, Vector<Double>) -> Vector<Double>
    public let jacobian: (Vector<Double>, Vector<Double>) -> Matrix<Double>
    
    public convenience init(
        dimensions: Dimensions,
        function: @escaping (Vector<Double>, Vector<Double>) -> Vector<Double>
    ) {
        self.init(dimensions: dimensions, function: function) { state, control in
            let jacobian = Jacobian(shape: (rows: dimensions.state, columns: dimensions.state))
            return jacobian.numeric(state: state) { function($0, control) }
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
    public func apply(state x: Vector<Double>, control u: Vector<Double>) -> Vector<Double> {
        return self.function(x, u)
    }
    
    public func jacobian(state x: Vector<Double>, control u: Vector<Double>) -> Matrix<Double> {
        return self.jacobian(x, u)
    }
    
    public func validate(for dimensions: Dimensions) throws {
        #if DEBUG
        let stateBefore: Vector<Double> = .init(rows: dimensions.state)
        let control: Vector<Double> = .init(rows: dimensions.control)
        
        let stateAfter = self.apply(state: stateBefore, control: control)
        
        if stateAfter.rows != dimensions.state {
            throw Error.invalid(
                message: "Expected output vector of \(dimensions.state) rows, found \(stateAfter.rows)"
            )
        }
        #endif
    }
}
