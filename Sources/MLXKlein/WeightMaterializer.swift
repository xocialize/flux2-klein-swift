// WeightMaterializer.swift — first-run download of the package's declared weight sources
// (v0.19.0 contract: the PACKAGE auto-materializes; the app only picks the models folder).
//
// Executes the `WeightSourcing` declaration on KleinConfiguration into the engine
// ModelStore layout (`<root>/<org>/<name>/…`), forwarding BYTE-ACCURATE progress to
// `WeightDownloadProgress` so the engine's PreparationMonitor surfaces a real, moving
// `.downloading(fraction:bytesPerSecond:)` phase.
//
// Files are streamed DIRECTLY to the store — deliberately not HubClient.downloadSnapshot,
// which (as of swift-huggingface 0.9.0) never delivers byte updates during a transfer
// (fraction sits at 0% for a whole multi-GB file) and double-stores every artifact
// through its own cache. HubClient is still used for the tree listing (auth + endpoint).
//
// Large files download as PARALLEL RANGED CHUNKS (the hf_transfer design): HF's resolve
// endpoint for xet-backed repos reconstructs through the CAS bridge at ~0.5 MB/s per
// cold connection (measured; classic LFS serves ~50 MB/s single-stream) — 8 ranged
// connections aggregate back to ~54 MB/s.

import Foundation
import HuggingFace
import MLXToolKit

enum WeightMaterializer {

    enum MaterializeError: Error, LocalizedError {
        case badRepoId(String)
        case noStoreRoot
        case httpStatus(String, Int)
        case sizeMismatch(String)
        var errorDescription: String? {
            switch self {
            case .badRepoId(let id): return "Malformed weight-source repo id '\(id)' (want org/name)."
            case .noStoreRoot:
                return "FLUX.2-klein has no local weights and no model store to download into — "
                    + "set an explicit snapshotPath or choose a models folder."
            case .httpStatus(let path, let code): return "Download of \(path) failed (HTTP \(code))."
            case .sizeMismatch(let path): return "Download of \(path) ended with the wrong size."
            }
        }
    }

    private static let parallelThreshold: Int64 = 64 << 20
    private static let chunkSize: Int64 = 64 << 20
    private static let workers = 8

    /// Download every `source` into `root` (ModelStore layout). Progress is
    /// byte-weighted and monotonic across ALL sources' files.
    static func materialize(_ sources: [WeightSource], into root: URL) async throws {
        let client = HubClient()   // env-detected endpoint + token; gated repos honor HF_TOKEN
        let store = ModelStore(root: root)

        // Enumerate everything first so the fraction denominator is global.
        struct Item { let repo: String; let revision: String; let path: String
                      let size: Int64; let destination: URL }
        var items: [Item] = []
        for source in sources {
            guard let repoId = Repo.ID(rawValue: source.repo),
                  let destination = store.directory(for: source.repo) else {
                throw MaterializeError.badRepoId(source.repo)
            }
            let revision = source.revision ?? "main"
            let entries = try await client.listFiles(in: repoId, revision: revision)
            for entry in entries where entry.type == .file {
                let globs = source.matching ?? []
                let matches = globs.isEmpty || globs.contains { fnmatch($0, entry.path, 0) == 0 }
                guard matches else { continue }
                let dest = destination.appendingPathComponent(entry.path)
                // Skip files already fully present (source-level resume).
                if let attrs = try? FileManager.default.attributesOfItem(atPath: dest.path),
                   (attrs[.size] as? Int64) == Int64(entry.size ?? -1) { continue }
                items.append(Item(repo: source.repo, revision: revision, path: entry.path,
                                  size: Int64(entry.size ?? 0), destination: dest))
            }
        }
        guard !items.isEmpty else { return }
        let totalBytes = max(items.reduce(0) { $0 + $1.size }, 1)

        // Deltas from every worker funnel through one counter, then BRIDGE back into
        // task context via AsyncStream: `WeightDownloadProgress.sink` is a TaskLocal,
        // so a report made on a URLSession delegate-queue thread reads an UNBOUND
        // sink and silently vanishes. The reporter child task inherits the caller's
        // binding; the delegate threads only yield byte totals.
        let started = Date()
        var streamContinuation: AsyncStream<Int64>.Continuation!
        let totals = AsyncStream<Int64>(bufferingPolicy: .bufferingNewest(1)) { streamContinuation = $0 }
        let continuation = streamContinuation!
        let reporter = Task {
            for await transferred in totals {
                WeightDownloadProgress.report(
                    fraction: min(Double(transferred) / Double(totalBytes), 1.0),
                    bytesPerSecond: Double(transferred) / max(Date().timeIntervalSince(started), 0.001))
            }
        }
        let counter = ByteCounter { transferred in
            continuation.yield(transferred)
        }
        do {
            for item in items {
                try await downloadItem(
                    repo: item.repo, revision: item.revision, path: item.path,
                    size: item.size, to: item.destination, counter: counter)
            }
        } catch {
            continuation.finish()
            await reporter.value
            throw error
        }
        continuation.finish()
        await reporter.value
        WeightDownloadProgress.report(fraction: 1.0, bytesPerSecond: nil)
    }

    // MARK: one file

    private static func downloadItem(
        repo: String, revision: String, path: String, size: Int64, to destination: URL,
        counter: ByteCounter
    ) async throws {
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)

        var request = URLRequest(url: URL(string:
            "https://huggingface.co/\(repo)/resolve/\(revision)/\(path)")!)
        if let token = hfToken() { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }

        let partial = destination.appendingPathExtension("partial")
        try? FileManager.default.removeItem(at: partial)
        FileManager.default.createFile(atPath: partial.path, contents: nil)

        do {
            if size >= parallelThreshold {
                try await downloadParallel(request: request, size: size,
                                           partial: partial, counter: counter)
            } else {
                let handle = try FileHandle(forWritingTo: partial)
                do {
                    try await stream(request, to: handle, expect206: false, counter: counter)
                    try handle.close()
                } catch {
                    try? handle.close()
                    throw error
                }
            }
        } catch {
            try? FileManager.default.removeItem(at: partial)
            throw error
        }
        if size > 0 {
            let final = ((try? FileManager.default.attributesOfItem(atPath: partial.path)[.size]) as? Int64) ?? 0
            guard final == size else { throw MaterializeError.sizeMismatch(path) }
        }
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: partial, to: destination)
    }

    /// Parallel ranged chunks written at their offsets into a preallocated file.
    private static func downloadParallel(
        request: URLRequest, size: Int64, partial: URL, counter: ByteCounter
    ) async throws {
        let pre = try FileHandle(forWritingTo: partial)
        try pre.truncate(atOffset: UInt64(size))
        try pre.close()

        var chunks: [(start: Int64, end: Int64)] = []
        var offset: Int64 = 0
        while offset < size {
            chunks.append((offset, Swift.min(offset + chunkSize, size) - 1))
            offset += chunkSize
        }
        let queue = ChunkQueue(chunks)
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0 ..< Swift.min(workers, chunks.count) {
                group.addTask {
                    while let chunk = queue.next() {
                        try Task.checkCancellation()
                        var req = request
                        req.setValue("bytes=\(chunk.start)-\(chunk.end)", forHTTPHeaderField: "Range")
                        let handle = try FileHandle(forWritingTo: partial)
                        do {
                            try handle.seek(toOffset: UInt64(chunk.start))
                            try await stream(req, to: handle, expect206: true, counter: counter)
                            try handle.close()
                        } catch {
                            try? handle.close()
                            throw error
                        }
                    }
                }
            }
            try await group.waitForAll()
        }
    }

    private final class ChunkQueue: @unchecked Sendable {
        private let lock = NSLock()
        private var chunks: [(start: Int64, end: Int64)]
        init(_ chunks: [(start: Int64, end: Int64)]) { self.chunks = chunks }
        func next() -> (start: Int64, end: Int64)? {
            lock.lock(); defer { lock.unlock() }
            return chunks.isEmpty ? nil : chunks.removeFirst()
        }
    }

    // MARK: transport

    /// Chunk-wise delegate streaming (didReceive Data), NOT URLSession.bytes: per-byte
    /// AsyncBytes iteration collapses to ~1 MB/s in unoptimized (-Onone) builds — a
    /// Debug app build made the download look stalled. Data chunks arrive at
    /// ~64 KB-1 MB regardless of optimization level.
    private static func stream(
        _ request: URLRequest, to handle: FileHandle, expect206: Bool, counter: ByteCounter
    ) async throws {
        let delegate = StreamingDelegate(handle: handle, counter: counter, expect206: expect206)
        let session = URLSession(configuration: .ephemeral, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                delegate.continuation = cont
                session.dataTask(with: request).resume()
            }
        } onCancel: {
            session.invalidateAndCancel()
        }
        try Task.checkCancellation()
    }

    private final class StreamingDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
        let handle: FileHandle
        let counter: ByteCounter
        let expect206: Bool
        var continuation: CheckedContinuation<Void, Error>?
        private var failedStatus: Int?

        init(handle: FileHandle, counter: ByteCounter, expect206: Bool) {
            self.handle = handle
            self.counter = counter
            self.expect206 = expect206
        }

        func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                        didReceive response: URLResponse,
                        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
            if let http = response as? HTTPURLResponse,
               !(200...299).contains(http.statusCode) || (expect206 && http.statusCode != 206) {
                failedStatus = http.statusCode
                completionHandler(.cancel)
            } else {
                completionHandler(.allow)
            }
        }

        func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
            try? handle.write(contentsOf: data)
            counter.add(Int64(data.count))
        }

        func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
            let cont = continuation
            continuation = nil
            if let failedStatus {
                cont?.resume(throwing: MaterializeError.httpStatus(
                    task.originalRequest?.url?.lastPathComponent ?? "?", failedStatus))
            } else if let error {
                cont?.resume(throwing: error)
            } else {
                cont?.resume()
            }
        }
    }

    /// Cross-worker byte total with throttled (~4/s) reporting.
    private final class ByteCounter: @unchecked Sendable {
        private let lock = NSLock()
        private var total: Int64 = 0
        private var lastReport = Date.distantPast
        private let onTotal: (Int64) -> Void
        init(onTotal: @escaping (Int64) -> Void) { self.onTotal = onTotal }
        func add(_ n: Int64) {
            lock.lock()
            total += n
            let snapshot = total
            let due = Date().timeIntervalSince(lastReport) > 0.25
            if due { lastReport = Date() }
            lock.unlock()
            if due { onTotal(snapshot) }
        }
    }

    /// HF token: env first, then the CLI token file (upstream convention).
    private static func hfToken() -> String? {
        if let t = ProcessInfo.processInfo.environment["HF_TOKEN"], !t.isEmpty { return t }
        let file = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/token")
        return (try? String(contentsOf: file, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
