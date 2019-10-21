import XCTest

import Surge
import StateSpaceModel

@testable import KalmanFilter

private func deg2rad(_ degree: Double) -> Double {
    return (degree / 180.0) * .pi
}

/// Modelled after:
/// https://github.com/balzer82/Kalman/blob/master/Extended-Kalman-Filter-CTRV.ipynb
final class NonlinearVelocityYawModelTests: XCTestCase {
    typealias MotionModel = ControllableNonlinearMotionModel
    typealias ObservationModel = LinearObservationModel

    let time: Double = 0.1 // time delta
    let velocity: Double = 5.0 // in m/s
    let yaw: Double = deg2rad(20.0) // yaw rate in radians/s^2

    let dimensions: Dimensions = .init(
        state: 5, // [position x, position y, h, velocity, yaw rate]
        control: 2, // [velocity, yaw rate]
        observation: 2 // [position x, position y]
    )

    lazy var motionModel: MotionModel = .init(dimensions: self.dimensions) { state, control in
        let (x, y, h) = (state[0], state[1], state[2]) // pos-x, pos-y, heading
        let (v, w) = (control[0], control[1]) // velocity, yaw-rate
        let t = self.time // delta time
        return [
            x + (v / w) * (sin(h + w * t) - sin(h)),
            y + (v / w) * (-cos(h + w * t) + cos(h)),
            h + w * t,
            v,
            w,
        ]
    }

    let observationModel: ObservationModel = .init(
        state: [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )

    lazy var processNoise: Matrix<Double> = {
        let acceleration = 1.0 // max expected acceleration in m/sec^2
        let yaw = 0.1 // max expected yaw in radians/s^2
        let qs: Matrix<Double> = [
            [acceleration * (0.5 * time * time)], // translation in m (double-integrated acceleration)
            [acceleration * (0.5 * time * time)], // translation in m (double-integrated acceleration)
            [yaw * time], // heading in radians/s (integrated of yaw)
            [acceleration * time], // velocity in m/s (integrated acceleration)
            [yaw * 1.0], // yaw in radians/s^2
        ]
        return (qs * qs.transposed()).squared()
    }()

    lazy var observationNoise: Matrix<Double> = Matrix.diagonal(
        rows: dimensions.observation,
        columns: dimensions.observation,
        repeatedValue: 2.0
    ).squared()

    lazy var predictor: KalmanPredictor = .init(
        motionModel: self.motionModel,
        processNoise: self.processNoise
    )

    lazy var updater: KalmanUpdater = .init(
        observationModel: self.observationModel,
        observationNoise: self.observationNoise
    )

    let initialState: Vector<Double> = [
        0.0, // Position X
        0.0, // Position Y
        0.0, // Heading
        0.0, // Velocity
        0.0, // Yaw Rate
    ]

    func estimate() -> (state: Vector<Double>, covariance: Matrix<Double>) {
        return (
            state: self.initialState,
            covariance: Matrix.diagonal(
                rows: 5,
                columns: 5,
                repeatedValue: 1.0
            )
        )
    }

    func filter(control: (Int) -> Vector<Double>) -> Double {
        let estimate = self.estimate()
        let initialState = self.initialState

        let sampleCount = 200
        let controls: [Vector<Double>] = (0..<sampleCount).map(control)

        let states = self.makeSignal(
            initial: initialState,
            controls: controls,
            model: self.motionModel,
            processNoise: self.processNoise
        )

        let observations: [Vector<Double>] = states.map { state in
            let observation: Vector<Double> = self.observationModel.apply(state: state)
            let standardNoise: Vector<Double> = Vector(gaussianRandom: self.dimensions.observation)
            let noise: Vector<Double> = self.observationNoise * standardNoise
            return observation + noise
        }

        var kalmanFilter = KalmanFilter(
            estimate: estimate,
            predictor: KalmanPredictor(
                motionModel: self.motionModel,
                processNoise: self.processNoise
            ),
            updater: KalmanUpdater(
                observationModel: self.observationModel,
                observationNoise: self.observationNoise
            )
        )

        let filteredStates: [Vector<Double>] = Swift.zip(controls, observations).map { argument in
            let (control, observation) = argument
            kalmanFilter.filter(control: control, observation: observation)
            return kalmanFilter.estimate.state
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
            let velocity = self.velocity
            let yaw = self.yaw
            return Vector([velocity, yaw])
        }

        XCTAssertLessThan(similarity, 25.0)
    }

    func testVariableModel() {
        let similarity = self.filter { i in
            let sine = sin(Double(i) * 0.1) * 0.5 + 0.5 // sine-wave from 0.0..1.0
            let cosine = cos(Double(i) * 0.1) * 0.5 + 0.5 // cosine-wave from 0.0..1.0
            let velocity = self.velocity * sine
            let yaw = self.yaw * cosine
            return Vector([velocity, yaw])
        }

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
        ("testConstantModel", testConstantModel),
        ("testVariableModel", testVariableModel),
    ]
}
