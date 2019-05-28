import Foundation

public class NoiseModel {
    /// Process noise matrix (aka `Q`)
    ///
    /// This matrix implies the process noise covariance.
    ///
    /// Default: zero matrix.
    public var process: Matrix<Double>
    
    /// Output noise matrix (aka `R`)
    ///
    /// This matrix implies the output error covariance,
    /// based on the amount of sensor noise.
    ///
    /// Default: zero matrix.
    public var output: Matrix<Double>
    
    public init(
        process: Matrix<Double>,
        output: Matrix<Double>
    ) {
        self.process = process
        self.output = output
    }
}
