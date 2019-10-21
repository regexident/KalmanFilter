import XCTest

import Surge
import StateSpace
import StateSpaceModel

@testable import KalmanFilter

final class LocalizationTests: XCTestCase {
    typealias MotionModel = ControllableLinearMotionModel<LinearMotionModel, LinearControlModel>
    typealias ObservationModel = NonlinearObservationModel

    let dimensions: Dimensions = Dimensions(
        state: 4, // [target position x, target position y, self position x, self position y]
        control: 2, // [self position x, self position y]
        observation: 1 // [distance]
    )

    lazy var motionModel: MotionModel = .init(
        a: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        b: [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    lazy var observationModel: ObservationModel = .init(dimensions: self.dimensions) { state in
        let targetPosition: Vector<Double> = [state[0], state[1]]
        let selfPosition: Vector<Double> = [state[2], state[3]]
        let delta = targetPosition - selfPosition
        let dist = delta.magnitude()
        return [dist]
    }

    lazy var processNoise: Matrix<Double> = .diagonal(
        rows: self.dimensions.state,
        columns: self.dimensions.state,
        repeatedValue: 0.0
    )

    lazy var observationNoise: Matrix<Double> = .diagonal(
        rows: self.dimensions.observation,
        columns: self.dimensions.observation,
        repeatedValue: 0.5
    )

    func testModel() {
        let initialState: Vector<Double> = [
            5.0, // target position x
            10.0, // target position y
            0.0, // self position x
            0.0, // self position y
        ]
        
        let estimate: KalmanEstimate = .init(
            state: initialState,
            covariance: Matrix.diagonal(
                rows: self.dimensions.state,
                columns: self.dimensions.state,
                repeatedValue: 1.0
            )
        )
        
        let interval = 0.05
        let sampleCount = 500
        let controls: [Vector<Double>] = (0..<sampleCount).map { i in
            let a = Double(i) * interval
            let r = 7.5
            let x = r * sin(a)
            let y = r * cos(a)
            return Vector([x, y])
        }
        
        let states = self.makeSignal(
            initial: estimate.state,
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
