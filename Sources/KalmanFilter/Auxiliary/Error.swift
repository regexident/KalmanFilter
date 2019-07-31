import Foundation

public enum MatrixError: Swift.Error {
    case invalidColumnCount(message: String)
    case invalidRowCount(message: String)
}
