import MLX
import XCTest

extension XCTestCase {
    /// Runs `body` with MLX pinned to the CPU. `Device.withDefaultDevice` alone only scopes the
    /// Swift-side default stream; mlx core internals (eval's synchronizer, compile) still use the
    /// global C++ default device, so a long CPU run left on a GPU default stalls a GPU command
    /// buffer until the GPU watchdog kills the process. The Swift default is latched first so
    /// switching the C++ default can't leave the whole process stuck on the CPU.
    func withMLXCPU(_ body: () -> Void) {
        _ = Device.defaultDevice()
        Device.setDefault(device: Device(.cpu))
        defer { Device.setDefault(device: Device(.gpu)) }
        Device.withDefaultDevice(Device(.cpu), body)
    }
}
