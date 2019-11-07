import XCTest

import Surge
import BayesFilter
import StateSpace
import StateSpaceModel

@testable import KalmanFilter

/// Modelled after:
/// https://github.com/balzer82/Kalman/blob/master/Kalman-Filter-CA.ipynb
final class LinearAccelerationModelTests: XCTestCase {
    typealias MotionModel = ControllableLinearMotionModel<LinearMotionModel, LinearControlModel>
    typealias ObservationModel = LinearObservationModel

    let time: Double = 0.1 // time delta
    let acceleration: (x: Double, y: Double) = (
        x: 2.0, // acceleration on x-axis
        y: 1.0 // acceleration on y-axis
    ) // in m/s^2

    lazy var dimensions = Dimensions(
        state: 6, // [position x, position y, velocity x, velocity y, acceleration x, acceleration y]
        control: 2, // [acceleration x, acceleration y]
        observation: 2 // [position x, position y]
    )

    lazy var motionModel: MotionModel = .init(
        a: [
            [1.0, 0.0, self.time, 0.0, 0.5 * self.time * self.time, 0.0],
            [0.0, 1.0, 0.0, self.time, 0.0, 0.5 * self.time * self.time],
            [0.0, 0.0, 1.0, 0.0, self.time, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, self.time],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        b: [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.5 * self.time * self.time, 0.0],
            [0.0, 0.5 * self.time * self.time],
            [self.time, 0.0],
            [0.0, self.time],
        ]
    )

    let observationModel: ObservationModel = .init(
        state: [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    lazy var processNoiseStdDeviations: Vector<Double> = {
        let acceleration = 1.0 // max expected acceleration in m/sec^2
        let time = self.time
        return [
            acceleration * 0.5 * time * time, // translation in m (double-integrated acceleration)
            acceleration * 0.5 * time * time, // translation in m (double-integrated acceleration)
            acceleration * time, // velocity in m/s (integrated acceleration)
            acceleration * time, // velocity in m/s (integrated acceleration)
            acceleration * 1.0, // acceleration in m/s^2
            acceleration * 1.0, // acceleration in m/s^2
        ]
    }()

    lazy var processNoiseCovariance: Matrix<Double> = {
        let variance = pow(self.processNoiseStdDeviations, 2.0)
        return Matrix.diagonal(
            rows: self.dimensions.state,
            columns: self.dimensions.state,
            scalars: variance
        )
    }()

    lazy var observationNoiseStdDeviations: Vector<Double> = {
        return [
            2.0, // position x
            2.0, // position y
        ]
    }()

    lazy var observationNoiseCovariance: Matrix<Double> = {
        let variance = pow(self.observationNoiseStdDeviations, 2.0)
        return Matrix.diagonal(
            rows: self.dimensions.observation,
            columns: self.dimensions.observation,
            scalars: variance
        )
    }()

    func filter(control: (Int) -> Vector<Double>) -> Double {
        let initialState: Vector<Double> = [
            0.0, // Position X
            0.0, // Position Y
            0.0, // Velocity X
            0.0, // Velocity Y
            0.0, // Acceleration X
            0.0, // Acceleration Y
        ]

        let estimate: KalmanEstimate = .init(
            state: initialState,
            covariance: Matrix.diagonal(
                rows: 6,
                columns: 6,
                repeatedValue: 1.0
            )
        )

        let sampleCount = 200
        let controls: [Vector<Double>] = (0..<sampleCount).map(control)

        let states = self.makeSignal(
            initial: initialState,
            controls: controls,
            model: self.motionModel,
            processNoise: self.processNoiseCovariance
        )

        let observations: [Vector<Double>] = states.map { state in
            let observation: Vector<Double> = self.observationModel.apply(state: state)
            let standardNoise: Vector<Double> = Vector.randomNormal(count: self.dimensions.observation)
            let noise: Vector<Double> = self.observationNoiseCovariance * standardNoise
            return observation + noise
        }

        let kalmanFilter = KalmanFilter(
            predictor: KalmanPredictor(
                motionModel: self.motionModel,
                processNoise: self.processNoiseCovariance
            ),
            updater: KalmanUpdater(
                observationModel: self.observationModel,
                observationNoise: self.observationNoiseCovariance
            )
        )

        var statefulKalmanFilter = StatefulKalmanFilter(
            estimate: estimate,
            wrapping: kalmanFilter
        )

        let filteredStates: [Vector<Double>] = Swift.zip(controls, observations).map { argument in
            let (control, observation) = argument
            statefulKalmanFilter.filter(control: control, observation: observation)
            return statefulKalmanFilter.estimate.state
        }

//        self.printSheetAndFail(
//            trueStates: states,
//            estimatedStates: filteredStates,
//            observations: observations
//        )

        let (similarity, _) = autoCorrelation(between: states, and: filteredStates, within: 10) { $0.distance(to: $1) }

        return similarity
    }

    func testConstantModel() {
        let similarity = self.filter { i in
            let x = self.acceleration.x
            let y = self.acceleration.y
            return Vector([x, y])
        }

        XCTAssertLessThan(similarity, 7.5)
    }

    func testVariableModel() {
        let similarity = self.filter { i in
            let sine = sin(Double(i) * 0.1) * 0.5 + 0.5 // sine-wave from 0.0..1.0
            let cosine = cos(Double(i) * 0.1) * 0.5 + 0.5 // cosine-wave from 0.0..1.0
            let x = self.acceleration.x * sine
            let y = self.acceleration.y * cosine
            return Vector([x, y])
        }

        XCTAssertLessThan(similarity, 7.5)
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
        ("testConstantModel", testConstantModel),
        ("testVariableModel", testVariableModel),
    ]
}
