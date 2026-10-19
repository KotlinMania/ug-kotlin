#if canImport(Testing)
import Testing
import Ug

@Suite("Ug Swift Export Tests")
struct UgExportTests {
    @Test("Shapes and layouts retain their dimensions across the Swift bridge")
    func testShapeLayout() {
        let shape = Shape(dims: [2, 3])
        #expect(shape.elemCount() == 6)
        #expect(shape.strideContiguous() == [3, 1])
        let layout = Layout.Companion.shared.fromShape(shape: shape)
        let transposed = layout.transpose()
        #expect(transposed.dims == [3, 2])
        #expect(transposed.numElements() == 6)
        #expect(WithDType.Companion.shared.F32.createArray(len: 2) == [0, 0])
        let error = UgError.Msg(message: "invalid tensor")
        #expect(error.withPath(path: "weights").message == "path: weights invalid tensor")
    }
}
#elseif canImport(XCTest)
import XCTest
import Ug

final class UgExportTests: XCTestCase {
    func testShapeLayout() {
        let shape = Shape(dims: [2, 3])
        XCTAssertEqual(shape.elemCount(), 6)
        XCTAssertEqual(shape.strideContiguous(), [3, 1])
        let layout = Layout.Companion.shared.fromShape(shape: shape)
        let transposed = layout.transpose()
        XCTAssertEqual(transposed.dims, [3, 2])
        XCTAssertEqual(transposed.numElements(), 6)
        XCTAssertEqual(WithDType.Companion.shared.F32.createArray(len: 2), [0, 0])
        let error = UgError.Msg(message: "invalid tensor")
        XCTAssertEqual(error.withPath(path: "weights").message, "path: weights invalid tensor")
    }
}
#endif
