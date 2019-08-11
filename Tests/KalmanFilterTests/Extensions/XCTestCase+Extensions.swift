import XCTest

import Surge

@testable import KalmanFilter

extension XCTestCase {
    internal func makeSignal<S>(
        initial initialState: Vector<Double>,
        controls: S,
        model: MotionModel,
        processNoise covariance: Matrix<Double>
    ) -> [Vector<Double>]
        where S: Sequence, S.Element == Vector<Double>
    {
        var signal: [Vector<Double>] = [initialState]
        var state = initialState
        
        for control in controls {
            state = model.apply(state: state, control: control)
            let standardNoise: Vector<Double> = Vector(gaussianRandom: state.dimensions)
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
        observations: [Vector<Double>]
    ) {
        assert(observations.count == trueStates.count)
        assert(observations.count == estimatedStates.count)
        
        guard observations.count > 0 else {
            return
        }
        
        let headerCellsTrueStates = (0..<trueStates[0].dimensions).map { "True \($0)" }.joined(separator: ",")
        let headerCellsEstimatedStates = (0..<estimatedStates[0].dimensions).map { "Estimated \($0)" }.joined(separator: ",")
        let headerCellsObservations = (0..<observations[0].dimensions).map { "Observation \($0)" }.joined(separator: ",")
        let headerRow = [
            "Time", headerCellsTrueStates, headerCellsEstimatedStates, headerCellsObservations
        ].joined(separator: ",")
        
        print(headerRow)
        
        for i in 0..<observations.count {
            let cellsTrueStates = trueStates[i].scalars.map { "\($0)" }.joined(separator: ",")
            let cellsEstimatedStates = estimatedStates[i].scalars.map { "\($0)" }.joined(separator: ",")
            let cellsObservations = observations[i].scalars.map { "\($0)" }.joined(separator: ",")
            let row = [
                "\(i)", cellsTrueStates, cellsEstimatedStates, cellsObservations
            ].joined(separator: ",")
            
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
