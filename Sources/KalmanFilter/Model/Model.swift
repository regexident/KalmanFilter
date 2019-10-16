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
    public var motion: MotionModel {
        willSet {
            #if DEBUG
            try! newValue.validate(for: self.dimensions)
            #endif
        }
    }
    
    /// The kalman filters's observation model (used for correction).
    public var observation: ObservationModel {
        willSet {
            #if DEBUG
            try! newValue.validate(for: self.dimensions)
            #endif
        }
    }
    
    /// The kalman filters's noise model (used for correction).
    public var noise: NoiseModel {
        willSet {
            #if DEBUG
            try! newValue.validate(for: self.dimensions)
            #endif
        }
    }
    
    public init(
        dimensions: Dimensions,
        motion: MotionModel,
        observation: ObservationModel,
        noise: NoiseModel
    ) {
        #if DEBUG
        try! motion.validate(for: dimensions)
        try! observation.validate(for: dimensions)
        try! noise.validate(for: dimensions)
        #endif
        
        self.dimensions = dimensions
        self.motion = motion
        self.observation = observation
        self.noise = noise
    }
}
