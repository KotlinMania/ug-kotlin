#if canImport(Testing)
import Testing
import Ug

@Suite("Ug Swift Export Tests")
struct UgExportTests {
    @Test("Ug swift module imported cleanly")
    func testSwiftModuleLoads() throws {
        #expect(Bool(true), "Ug swift module imported cleanly")
    }
}
#elseif canImport(XCTest)
import XCTest
import Ug

final class UgExportTests: XCTestCase {
    func testSwiftModuleLoads() throws {
        XCTAssertTrue(true, "Ug swift module imported cleanly")
    }
}
#endif
