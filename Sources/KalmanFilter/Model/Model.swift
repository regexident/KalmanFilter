import Foundation

public class Model {
    /// The kalman filters's model dimensions.
    public let dimensions: Dimensions
    
    /// The kalman filters's motion model (used for prediction).
    public var motionModel: MotionModel
    
    /// The kalman filters's observation model (used for correction).
    public var observationModel: ObservationModel
    
    /// The kalman filters's noise model (used for correction).
    public var noiseModel: NoiseModel
    
    public init(
        dimensions: Dimensions,
        motionModel: MotionModel,
        observationModel: ObservationModel,
        noiseModel: NoiseModel
    ) {
        self.dimensions = dimensions
        self.motionModel = motionModel
        self.observationModel = observationModel
        self.noiseModel = noiseModel
    }
}
