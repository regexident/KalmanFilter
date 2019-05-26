import XCTest

@testable import KalmanFilter

final class LocalizationTests: XCTestCase {
    let configuration: Configuration = {
        let dimensions = Dimensions(
            state: 4, // [target position x, target position y, self position x, self position y]
            input: 2, // [self position x, self position y]
            output: 1 // [distance]
        )
        
        let initialState: Vector = [
            5.0, // Target Position X
            10.0, // Target Position Y
            0.0, // Self Position X
            0.0, // Self Position Y
        ]
        
        let motionModel = StaticMatrixMotionModel(
            a: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ], b: [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        
        let observationModel = FunctionObservationModel(output: 1) { state in
            let targetPosition: Vector<Double> = [state[0], state[1]]
            let selfPosition: Vector<Double> = [state[2], state[3]]
            let delta = targetPosition - selfPosition
            let dist = delta.magnitude
            return [dist]
        }
        
        let processNoiseCovariance = Matrix(diagonal: 0.0, size: 4)
        
        let outputNoiseVariance = 2.0
        let outputNoiseCovariance = Matrix(diagonal: outputNoiseVariance, size: dimensions.output).squared()
        
        let estimateCovariance: Matrix<Double> = Matrix(diagonal: 1.0, size: dimensions.state)
        
        return Configuration(dimensions: dimensions) { config in
            config.state = initialState
            config.motionModel = motionModel
            config.observationModel = observationModel
            config.estimateCovariance = estimateCovariance
            config.processNoiseCovariance = processNoiseCovariance
            config.outputNoiseCovariance = outputNoiseCovariance
        }
    }()
    
    func testModel() {
        let configuration = self.configuration
        
        let interval = 0.1
        let sampleCount = 200
        let inputs: [Vector<Double>] = (0..<sampleCount).map { i in
            let a = Double(i) * interval
            let r = 7.5
            let x = r * sin(a)
            let y = r * cos(a)
            return Vector(column: [x, y])
        }
        
        let states = self.makeSignal(
            initial: configuration.state,
            inputs: inputs,
            model: configuration.motionModel,
            processNoise: configuration.processNoiseCovariance
        )
        
        let outputs: [Vector<Double>] = states.map { state in
            let output: Vector<Double> = configuration.observationModel.apply(state: state)
            let standardNoise: Vector<Double> = Vector(gaussianRandom: configuration.dimensions.output)
            let noise: Vector<Double> = configuration.outputNoiseCovariance * standardNoise
            return output + noise
        }
        
        let kalmanFilter = KalmanFilter(configuration)

        let filteredStates: [Vector<Double>] = Swift.zip(inputs, outputs).map { input, output in
            let filteredState = kalmanFilter.filter(output: output, input: input)
            return filteredState
        }
        
//        self.printSheet(unfiltered: states, filtered: filteredStates, measured: outputs)
        
        let (similarity, _) = autoCorrelation(between: states, and: filteredStates, within: 10) { $0.distance(to: $1) }

        XCTAssertLessThan(similarity, 5.0)
    }
    
    private func printSheet(
        unfiltered: [Vector<Double>],
        filtered: [Vector<Double>],
        measured: [Vector<Double>]
    ) {
        self.printSheet(unfiltered2D: unfiltered, filtered2D: filtered, measured1D: measured)
        
        XCTFail("Printing found in test")
    }
    
    static var allTests = [
        ("testModel", testModel),
    ]
}
