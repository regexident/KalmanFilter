import Foundation

public struct Dimensions {
    public let state: Int
    public let input: Int
    public let output: Int
    
    public init(state: Int, input: Int, output: Int) {
        assert(state >= 1)
        assert(input >= 1)
        assert(output >= 1)
        
        self.state = state
        self.input = input
        self.output = output
    }
    
    public init(uniform: Int) {
        assert(uniform >= 1)
        
        self.init(state: uniform, input: uniform, output: uniform)
    }
}

extension Dimensions: CustomStringConvertible {
    public var description: String {
        return "{ state: \(self.state), input: \(self.input), output: \(self.output) }"
    }
}
