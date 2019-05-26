import XCTest

import KalmanFilter

extension XCTestCase {
    internal func makeSignal<S>(
        initial initialState: Vector<Double>,
        inputs: S,
        model: MotionModel,
        processNoise covariance: Matrix<Double>
    ) -> [Vector<Double>]
        where S: Sequence, S.Element == Vector<Double>
    {
        var signal: [Vector<Double>] = [initialState]
        var state = initialState
        
        for input in inputs {
            state = model.apply(state: state, input: input)
            let standardNoise: Vector<Double> = Vector(gaussianRandom: state.rows)
            let noise: Vector<Double> = covariance * standardNoise
            signal.append(state + noise)
        }
        
        if signal.count > 1 {
            signal.removeLast()
        }
        
        return signal
    }
    
    internal func printSheet(
        unfiltered2D: [Vector<Double>],
        filtered2D: [Vector<Double>],
        measured2D: [Vector<Double>]
    ) {
        assert(measured2D.count == unfiltered2D.count)
        assert(measured2D.count == filtered2D.count)
        
        for i in 0..<measured2D.count {
            let (unfiltered, filtered, measured) = (unfiltered2D[i], filtered2D[i], measured2D[i])
            
            let (x1, y1) = (unfiltered[0], unfiltered[1])
            let (x2, y2) = (filtered[0], filtered[1])
            let (x3, y3) = (measured[0], measured[1])
            
            print("\(x1),\(y1),\(x2),\(y2),\(x3),\(y3)")
        }
    }
    
    internal func printSheet(
        unfiltered2D: [Vector<Double>],
        filtered2D: [Vector<Double>],
        measured1D: [Vector<Double>]
    ) {
        assert(measured1D.count == unfiltered2D.count)
        assert(measured1D.count == filtered2D.count)
        
        for i in 0..<measured1D.count {
            let (unfiltered, filtered, measured) = (unfiltered2D[i], filtered2D[i], measured1D[i])
            
            let (x1, y1) = (unfiltered[0], unfiltered[1])
            let (x2, y2) = (filtered[0], filtered[1])
            let z3 = (measured[0])
            
            print("\(x1),\(y1),\(x2),\(y2),\(z3)")
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
