import XCTest

import Surge
import BayesFilter
import StateSpace
import StateSpaceModel

@testable import KalmanFilter

final class LandmarkLocalizationTests: XCTestCase {
    typealias MotionModel = ControllableLinearMotionModel<LinearMotionModel, LinearControlModel>
    typealias ObservationModel = NonlinearObservationModel

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

    let dimensions: Dimensions = .init(
        state: 4, // [position x, position y, velocity x, velocity y]
        control: 2, // [velocity x, velocity y]
        observation: 1 // [distance to landmark]
    )

    lazy var motionModel: MotionModel = .init(
        a: [
            [1.0, 0.0, self.time, 0.0],
            [0.0, 1.0, 0.0, self.time],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        b: [
            [0.0, 0.0],
            [0.0, 0.0],
            [self.time, 0.0],
            [0.0, self.time],
        ]
    )

    func observationModel(landmark: Landmark) -> ObservationModel {
        return .init(dimensions: self.dimensions) { state in
            let targetPosition: Vector<Double> = [state[0], state[1]]
            let landmarkPosition: Vector<Double> = landmark.location
            let dist = targetPosition.distance(to: landmarkPosition)
            return [dist]
        }
    }

    lazy var processNoise: Matrix<Double> = {
        let t = self.time
        let accel = 0.25 // max expected acceleration in m/sec^2
        let qs: Matrix<Double> = [
            [accel * 0.5 * t * t], // translation in m (double-integrated acceleration)
            [accel * 0.5 * t * t], // translation in m (double-integrated acceleration)
            [accel * t], // velocity in m/s (integrated acceleration)
            [accel * t], // velocity in m/s (integrated acceleration)
        ]
        return pow((qs * transpose(qs)), 2.0)
    }()

    lazy var observationNoise: Matrix<Double> = .diagonal(
        rows: self.dimensions.observation,
        columns: self.dimensions.observation,
        repeatedValue: 0.5
    )

    func filter(control: (Int) -> Vector<Double>) -> Double {
        let initialState: Vector<Double> = [
            0.0, // target position X
            0.0, // target position Y
            0.0, // target velocity x
            0.0, // target velocity y
        ]

        let estimate: KalmanEstimate = .init(
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

        let sampleCount = 200
        let controls: [Vector<Double>] = (0..<sampleCount).map(control)

        let states = self.makeSignal(
            initial: initialState,
            controls: controls,
            model: motionModel,
            processNoise: self.processNoise
        )

        let observations: [[Vector<Double>]] = states.map { state in
            landmarks.map { landmark in
                let observationModel = self.observationModel(landmark: landmark)
                let observation: Vector<Double> = observationModel.apply(state: state)
                let standardNoise: Vector<Double> = Vector.randomNormal(count: self.dimensions.observation)
                let noise: Vector<Double> = self.observationNoise * standardNoise
                let noisyObservation = observation + noise
                return noisyObservation
            }
        }

        let kalmanFilter = KalmanFilter(
            predictor: KalmanPredictor(
                motionModel: self.motionModel,
                processNoise: self.processNoise
            ),
            updater: MultiModalKalmanUpdater { landmark in
                return KalmanUpdater(
                    observationModel: self.observationModel(landmark: landmark),
                    observationNoise: self.observationNoise
                )
            }
        )

        var statefulKalmanFilter = Estimateful(
            estimate: estimate,
            wrapping: kalmanFilter
        )

        let filteredStates: [Vector<Double>] = Swift.zip(controls, observations).map { argument in
            let (control, observations) = argument
            for (landmark, observation) in Swift.zip(landmarks, observations) {
                let _ = statefulKalmanFilter.filter(
                    control: control,
                    observation: MultiModal(model: landmark, value: observation)
                )
            }
            return statefulKalmanFilter.estimate.state
        }

//        self.printSheetAndFail(
//            trueStates: states,
//            estimatedStates: filteredStates,
//            observations: observations.map { observations in
//                Vector(observations.map { $0[0] })
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
        observations: [Vector<Double>]? = nil
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
