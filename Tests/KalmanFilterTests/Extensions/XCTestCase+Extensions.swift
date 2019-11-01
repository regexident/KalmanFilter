import XCTest

import Surge
import StateSpace
import StateSpaceModel

@testable import KalmanFilter

extension XCTestCase {
    internal func makeSignal<Controls, MotionModel>(
        initial initialState: Vector<Double>,
        controls: Controls,
        model: MotionModel,
        processNoise covariance: Matrix<Double>
    ) -> [Vector<Double>]
        where Controls: Sequence, Controls.Element == Vector<Double>,
              MotionModel: Statable & ControllableMotionModelProtocol,
              MotionModel.State == Vector<Double>,
              MotionModel.Control == Vector<Double>
    {
        var signal: [Vector<Double>] = [initialState]
        var state = initialState
        
        for control in controls {
            state = model.apply(state: state, control: control)
            let standardNoise: Vector<Double> = Vector.randomNormal(count: state.dimensions)
            let noise: Vector<Double> = covariance * standardNoise
            signal.append(state + noise)
        }
        
        if signal.count > 1 {
            signal.removeLast()
        }
        
        return signal
    }

    internal func makeSignal<MotionModel>(
        initial initialState: Vector<Double>,
        count: Int,
        model: MotionModel,
        processNoise covariance: Matrix<Double>
    ) -> [Vector<Double>]
        where MotionModel: Statable & UncontrollableMotionModelProtocol,
              MotionModel.State == Vector<Double>
    {
        var signal: [MotionModel.State] = [initialState]
        var state = initialState

        for _ in 0..<count {
            state = model.apply(state: state)
            let standardNoise: Vector<Double> = Vector.randomNormal(count: state.dimensions)
            let noise: Vector<Double> = covariance * standardNoise
            signal.append(state + noise)
        }

        if signal.count > 1 {
            signal.removeLast()
        }

        return signal
    }
    
    internal func printSheet(
        trueStates: [Vector<Double>],
        estimatedStates: [Vector<Double>],
        observations: [Vector<Double>]? = nil
    ) {
        let sampleCount = trueStates.count

        assert(estimatedStates.count == sampleCount)

        if let observations = observations {
            assert(observations.count == sampleCount)

            guard observations.count > 0 else {
                return
            }
        }

        let headerCellsTrueStates = (0..<trueStates[0].dimensions).map { "True \($0)" }.joined(separator: ",")
        let headerCellsEstimatedStates = (0..<estimatedStates[0].dimensions).map { "Estimated \($0)" }.joined(separator: ",")
        let headerCellsEstimationFitness = "Fitness"
        let headerCellsObservations = observations.map { observations in
            (0..<observations[0].dimensions).map { "Observation \($0)" }.joined(separator: ",")
        }
        let headerRow = [
            "Time", headerCellsTrueStates, headerCellsEstimatedStates, headerCellsEstimationFitness, headerCellsObservations
        ].compactMap { $0 }.joined(separator: ",")

        print(headerRow)

        for i in 0..<sampleCount {
            let cellsTrueStates = trueStates[i].scalars.map { "\($0)" }.joined(separator: ",")
            let cellsEstimatedStates = estimatedStates[i].scalars.map { "\($0)" }.joined(separator: ",")
            let cellsEstimationFitness = "\(trueStates[i].distance(to: estimatedStates[i]))"

            let cellsObservations = observations.map { observations in
                observations[i].scalars.map { "\($0)" }.joined(separator: ",")
            }
            let row = [
                "\(i)", cellsTrueStates, cellsEstimatedStates, cellsEstimationFitness, cellsObservations
            ].compactMap { $0 }.joined(separator: ",")

            print(row)
        }
    }
    
    internal func autoCorrelation<L, R>(
        between lhs: [L],
        and rhs: [R],
        within window: Int,
        kernel: (L, R) -> Double) -> (Double, (Int, Int)
    ) {
        assert(window < lhs.count)
        assert(window < rhs.count)
        
        var offsets: [(Int, Int)] = [(0, 0)]
        
        if window > 0 {
            for offset in 1...window {
                offsets.append((0, offset))
                offsets.append((offset, 0))
            }
        }
        
        var bestScore: Double = .greatestFiniteMagnitude
        var bestOffsets: (Int, Int) = (0, 0)
        
        for (l, r) in offsets {
            let lhs = lhs[l...]
            let rhs = rhs[r...]
            let count = min(lhs.count, rhs.count)
            let score = Swift.zip(lhs, rhs).reduce(0.0) { sum, pair in
                let (lhs, rhs) = pair
                let error = kernel(lhs, rhs)
                return sum + (error * error)
            } / Double(count)
            
            if score < bestScore {
                bestScore = score
                bestOffsets = (l, r)
            }
        }
        
        return (bestScore, bestOffsets)
    }
}
