import Foundation

public struct Dimensions {
    public let state: Int
    public let control: Int
    public let observation: Int
    
    public init(state: Int, control: Int, observation: Int) {
        assert(state >= 1)
        assert(control >= 1)
        assert(observation >= 1)
        
        self.state = state
        self.control = control
        self.observation = observation
    }
    
    public init(uniform: Int) {
        assert(uniform >= 1)
        
        self.init(state: uniform, control: uniform, observation: uniform)
    }
}

extension Dimensions: CustomStringConvertible {
    public var description: String {
        return "{ state: \(self.state), control: \(self.control), observation: \(self.observation) }"
    }
}
