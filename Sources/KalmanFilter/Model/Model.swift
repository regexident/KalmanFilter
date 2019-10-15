import Foundation

public class Model {
    public struct Error: Swift.Error {
        public let motion: LinearMotionModel.Error?
        public let observation: LinearObservationModel.Error?
        public let noise: NoiseModel.Error?
    }
    
    /// The kalman filters's model dimensions.
    public let dimensions: Dimensions
    
    /// The kalman filters's motion model (used for prediction).
    public var motionModel: MotionModel {
        willSet {
            #if DEBUG
            try! newValue.validate(for: self.dimensions)
            #endif
        }
    }
    
    /// The kalman filters's observation model (used for correction).
    public var observationModel: ObservationModel {
        willSet {
            #if DEBUG
            try! newValue.validate(for: self.dimensions)
            #endif
        }
    }
    
    /// The kalman filters's noise model (used for correction).
    public var noiseModel: NoiseModel {
        willSet {
            #if DEBUG
            try! newValue.validate(for: self.dimensions)
            #endif
        }
    }
    
    public init(
        dimensions: Dimensions,
        motionModel: MotionModel,
        observationModel: ObservationModel,
        noiseModel: NoiseModel
    ) {
        #if DEBUG
        try! motionModel.validate(for: dimensions)
        try! observationModel.validate(for: dimensions)
        try! noiseModel.validate(for: dimensions)
        #endif
        
        self.dimensions = dimensions
        self.motionModel = motionModel
        self.observationModel = observationModel
        self.noiseModel = noiseModel
    }
}
