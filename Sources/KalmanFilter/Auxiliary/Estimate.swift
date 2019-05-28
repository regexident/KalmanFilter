import Foundation

public struct Estimate {
    /// State vector (aka `x`)
    ///
    /// This vector implies the current state.
    ///
    /// Default: zero vector.
    public var state: Vector<Double>
    
    /// Estimate covariance matrix (aka `P`, or sometimes `Σ`)
    ///
    /// This matrix implies the evaluation noise covariance.
    ///
    /// Default: identity matrix.
    ///
    /// Initialization: unless more detailed domain-specific knowledge is available
    /// a good starting-point is: `P = variance * I` where `I` is the identity matrix.
    public var covariance: Matrix<Double>
    
    public init(state: Vector<Double>, covariance: Matrix<Double>) {
        self.state = state
        self.covariance = covariance
    }
}
