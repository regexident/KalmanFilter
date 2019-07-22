import Foundation

public struct Dimensions {
    public let state: Int
    public let control: Int
    public let output: Int
    
    public init(state: Int, control: Int, output: Int) {
        assert(state >= 1)
        assert(control >= 1)
        assert(output >= 1)
        
        self.state = state
        self.control = control
        self.output = output
    }
    
    public init(uniform: Int) {
        assert(uniform >= 1)
        
        self.init(state: uniform, control: uniform, output: uniform)
    }
}

extension Dimensions: CustomStringConvertible {
    public var description: String {
        return "{ state: \(self.state), control: \(self.control), output: \(self.output) }"
    }
}
