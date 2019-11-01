import Foundation

extension Float {
    internal static func gaussianRandom(mean: Float, stdDeviation: Float, samples: Int = 5) -> Float {
        let random = self.gaussianRandom()
        
        return mean + (random * stdDeviation)
    }
    
    internal static func gaussianRandom() -> Float {
        let a = Float.random(in: 0...1.0)
        let b = Float.random(in: 0...1.0)
        let c = Float.random(in: 0...1.0)
        
        let d = abs(-2.0 * log(a))
        let e = 2.0 * .pi * b
        
        let f: Float
        if c < 0.5 {
            f = sqrt(d) * sin(e)
        } else {
            f = sqrt(d) * cos(e)
        }
        
        return min(1.0, max(-1.0, f))
    }
}

extension Double {
    internal static func gaussianRandom(mean: Double, stdDeviation: Double, samples: Int = 5) -> Double {
        let random = self.gaussianRandom()

        return mean + (random * stdDeviation)
    }

    internal static func gaussianRandom() -> Double {
        let a = Double.random(in: 0...1.0)
        let b = Double.random(in: 0...1.0)
        let c = Double.random(in: 0...1.0)

        let d = abs(-2.0 * log(a))
        let e = 2.0 * .pi * b

        let f: Double
        if c < 0.5 {
            f = sqrt(d) * sin(e)
        } else {
            f = sqrt(d) * cos(e)
        }

        return min(1.0, max(-1.0, f))
    }
}
