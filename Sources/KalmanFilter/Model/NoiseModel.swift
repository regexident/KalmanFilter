import Foundation

public class NoiseModel {
    /// Process noise matrix (aka `Q`)
    ///
    /// This matrix implies the process noise covariance.
    ///
    /// Default: zero matrix.
    public var process: Matrix<Double>
    
    /// observation noise matrix (aka `R`)
    ///
    /// This matrix implies the observation error covariance,
    /// based on the amount of sensor noise.
    ///
    /// Default: zero matrix.
    public var observation: Matrix<Double>
    
    public init(
        process: Matrix<Double>,
        observation: Matrix<Double>
    ) {
        self.process = process
        self.observation = observation
    }
}
