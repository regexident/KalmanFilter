import Foundation

import Surge

public class NoiseModel {
    public enum Error: Swift.Error {
        case process(MatrixError)
        case observation(MatrixError)
    }
    
    /// Process noise matrix (aka `Q`)
    ///
    /// This matrix implies the process noise covariance.
    ///
    /// Default: zero matrix.
    public var process: Matrix<Double>
    
    /// Observation noise matrix (aka `R`)
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
    
    /// Validate the model for a given dimensional environment
    ///
    /// - Parameters:
    ///   - dimensions: the environment's dimensions
    public func validate(for dimensions: Dimensions) throws {
        guard self.process.columns == dimensions.state else {
            throw Error.process(.invalidColumnCount(
                message: "Expected \(dimensions.state) columns in `self.process`, found \(self.process.columns)"
            ))
        }
        
        guard self.process.rows == dimensions.state else {
            throw Error.process(.invalidRowCount(
                message: "Expected \(dimensions.state) columns in `self.process`, found \(self.process.rows)"
            ))
        }
        
        guard self.observation.columns == dimensions.observation else {
            throw Error.observation(.invalidColumnCount(
                message: "Expected \(dimensions.observation) columns in `self.observation`, found \(self.observation.columns)"
            ))
        }
        
        guard self.observation.rows == dimensions.observation else {
            throw Error.observation(.invalidRowCount(
                message: "Expected \(dimensions.observation) columns in `self.observation`, found \(self.observation.rows)"
            ))
        }
    }
}
