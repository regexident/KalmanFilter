import XCTest

@testable import KalmanFilter

final class LandmarkLocalizationTests: XCTestCase {
    struct Landmark: Hashable {
        let location: Vector<Double>
        let identifier: UUID

        public init(location: Vector<Double>, identifier: UUID = .init()) {
            self.location = location
            self.identifier = identifier
        }
        
        static func == (lhs: Landmark, rhs: Landmark) -> Bool {
            return lhs.identifier == rhs.identifier
        }
        
        func hash(into hasher: inout Hasher) {
            self.identifier.hash(into: &hasher)
        }
    }
    
    let dimensions = Dimensions(
        state: 2, // [target position x, target position y]
        control: 2, // [translation x, translation y]
        observation: 1 // [distance to landmark]
    )
    
    func motionModel(dimensions: Dimensions) -> MotionModel {
        return LinearMotionModel(
            state: [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            control: [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
    }
    
    func noiseModel(dimensions: Dimensions) -> NoiseModel {
        return NoiseModel(
            process: Matrix(diagonal: 0.0, size: dimensions.state),
            observation: {
                return Matrix(
                    diagonal: 0.5,
                    size: self.dimensions.observation
                ).squared()
            }()
        )
    }
    
    func observationModel(landmark: Landmark, dimensions: Dimensions) -> ObservationModel {
        NonlinearObservationModel(dimensions: dimensions) { state in
            let targetPosition: Vector<Double> = state
            let landmarkPosition: Vector<Double> = landmark.location
            let delta = targetPosition - landmarkPosition
            let dist = delta.magnitude
            return [dist]
        }
    }
    
    func model(landmark: Landmark, dimensions: Dimensions) -> Model {
        let motionModel = self.motionModel(dimensions: dimensions)
        let observationModel = self.observationModel(landmark: landmark, dimensions: dimensions)
        let noiseModel = self.noiseModel(dimensions: dimensions)

        return try! Model(
            dimensions: dimensions,
            motionModel: motionModel,
            observationModel: observationModel,
            noiseModel: noiseModel
        )
    }

    func testModel() {
        let initialState: Vector<Double> = [
            5.0, // Target Position X
            10.0, // Target Position Y
        ]
        
        let estimate: (state: Vector<Double>, covariance: Matrix<Double>) = (
            state: [
                7.0, // Target Position X
                7.0, // Target Position Y
            ],
            covariance: Matrix(diagonal: 10.0, size: self.dimensions.state)
        )
        
        let landmarks: [Landmark] = [
            Landmark(location: [-1.0, -1.0]),
            Landmark(location: [-1.0, 1.0]),
            Landmark(location: [1.0, -1.0]),
            Landmark(location: [1.0, 1.0]),
        ]

        let sampleCount = 500
        let controls: [Vector<Double>] = (0..<sampleCount).map { i in
            return Vector(column: [0.0, 0.0])
        }
        
        let motionModel = self.motionModel(dimensions: self.dimensions)
        let noiseModel = self.noiseModel(dimensions: self.dimensions)

        let states = self.makeSignal(
            initial: initialState,
            controls: controls,
            model: motionModel,
            processNoise: noiseModel.process
        )

        let observations: [[Vector<Double>]] = states.map { state in
            landmarks.map { landmark in
                let observationModel = self.observationModel(landmark: landmark, dimensions: self.dimensions)
                
                let observation: Vector<Double> = observationModel.apply(state: state)
                let standardNoise: Vector<Double> = Vector(gaussianRandom: self.dimensions.observation)
                let noise: Vector<Double> = noiseModel.observation * standardNoise
                let noisyObservation = observation + noise
                return noisyObservation
            }
        }

        var kalmanFilter: ContextSwitchingKalmanFilter<Landmark> = .init(
            dimensions: self.dimensions,
            estimate: estimate
        ) { landmark, dimensions, estimate in
            let model = self.model(landmark: landmark, dimensions: dimensions)
            return KalmanFilter(estimate: estimate, model: model)
        }
        
        let filteredStates: [Vector<Double>] = Swift.zip(controls, observations).map { argument in
            let (control, observations) = argument
            for (landmark, observation) in Swift.zip(landmarks, observations) {
                let _ = kalmanFilter.filter(
                    observation: observation,
                    control: Contextual(context: landmark, payload: control)
                )
            }
            return kalmanFilter.estimate.state
        }
        
        let flattenedObservations = observations.map { observations in
            Vector(column: observations.map { $0[0] })
        }

        // self.printSheetAndFail(trueStates: states, estimatedStates: filteredStates, observations: flattenedObservations)

        let (similarity, _) = autoCorrelation(between: states, and: filteredStates, within: 10) { $0.distance(to: $1) }

        XCTAssertLessThan(similarity, 5.0)
    }

    private func printSheetAndFail(
        trueStates: [Vector<Double>],
        estimatedStates: [Vector<Double>],
        observations: [Vector<Double>]
    ) {
        self.printSheet(trueStates: trueStates, estimatedStates: estimatedStates, observations: observations)

        XCTFail("Printing found in test")
    }

    static var allTests = [
        ("testModel", testModel),
    ]
}
