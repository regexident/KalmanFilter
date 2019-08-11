import XCTest

import Surge

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
    
    let time: Double = 1.0 // time delta in seconds
    
    let velocity: (x: Double, y: Double) = (x: 0.125, y: 0.125) // in meters per second
    
    let dimensions = Dimensions(
        state: 4, // [position x, position y, velocity x, velocity y]
        control: 2, // [velocity x, velocity y]
        observation: 1 // [distance to landmark]
    )
    
    func motionModel(dimensions: Dimensions) -> MotionModel {
        let t = self.time
        return LinearMotionModel(
            state: [
                [1.0, 0.0, t, 0.0],
                [0.0, 1.0, 0.0, t],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            control: [
                [0.0, 0.0],
                [0.0, 0.0],
                [t, 0.0],
                [0.0, t],
            ]
        )
    }
    
    func noiseModel(dimensions: Dimensions) -> NoiseModel {
        return NoiseModel(
            process: {
                let t = self.time
                let accel = 0.25 // max expected acceleration in m/sec^2
                let qs: Matrix = [
                    [accel * 0.5 * t * t], // translation in m (double-integrated acceleration)
                    [accel * 0.5 * t * t], // translation in m (double-integrated acceleration)
                    [accel * t], // velocity in m/s (integrated acceleration)
                    [accel * t], // velocity in m/s (integrated acceleration)
                ]
                return (qs * qs.transposed()).squared()
            }(),
            observation: Matrix.diagonal(
                rows: self.dimensions.observation,
                columns: self.dimensions.observation,
                repeatedValue: 0.5 // 2.0,
            ).squared()
        )
    }
    
    func observationModel(landmark: Landmark, dimensions: Dimensions) -> ObservationModel {
        NonlinearObservationModel(dimensions: dimensions) { state in
            let targetPosition: Vector<Double> = [state[0], state[1]]
            let landmarkPosition: Vector<Double> = landmark.location
            let dist = targetPosition.distance(to: landmarkPosition)
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

    func filter(control: (Int) -> Vector<Double>) -> Double {
        let initialState: Vector<Double> = [
            0.0, // target position X
            0.0, // target position Y
            0.0, // target velocity x
            0.0, // target velocity y
        ]
        
        let estimate: (state: Vector<Double>, covariance: Matrix<Double>) = (
            state: initialState,
            covariance: Matrix.diagonal(
                rows: self.dimensions.state,
                columns: self.dimensions.state,
                repeatedValue: 10.0
            )
        )
        
        let landmarks: [Landmark] = [
            Landmark(location: [-10.0, -10.0]),
            Landmark(location: [-10.0, 10.0]),
            Landmark(location: [10.0, -10.0]),
            Landmark(location: [10.0, 10.0]),
        ]

        let sampleCount = 500
        let controls: [Vector<Double>] = (0..<sampleCount).map(control)
        
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
        
//        self.printSheetAndFail(
//            trueStates: states,
//            estimatedStates: filteredStates,
//            observations: observations.map { observations in
//                Vector(column: observations.map { $0[0] })
//            }
//        )

        let (similarity, _) = autoCorrelation(between: states, and: filteredStates, within: 10) { $0.distance(to: $1) }

        return similarity
    }
    
    func testStaticModel() {
        let similarity = self.filter { i in
            return Vector([0.0, 0.0])
        }
        
        XCTAssertLessThan(similarity, 0.5)
    }
    
    func testConstantModel() {
        let similarity = self.filter { i in
            let x = self.velocity.x
            let y = self.velocity.y
            return Vector([x, y])
        }
        
        XCTAssertLessThan(similarity, 0.5)
    }
    
    func testVariableModel() {
        let similarity = self.filter { i in
            let waveX = sin(Double(i) * 0.1)
            let waveY = cos(Double(i) * 0.1)
            let x = self.velocity.x * waveX
            let y = self.velocity.y * waveY
            return Vector([x, y])
        }
        
        XCTAssertLessThan(similarity, 10.0)
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
        ("testStaticModel", testStaticModel),
        ("testConstantModel", testConstantModel),
        ("testVariableModel", testVariableModel),
    ]
}
