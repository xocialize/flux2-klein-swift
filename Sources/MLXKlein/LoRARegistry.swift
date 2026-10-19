// Curated registry of FLUX.2-klein-4B LoRAs (the Klein analog of xocialize/ltx-lora-registry).
//
// The manifest (Resources/klein-lora-registry.json) references each adapter's HuggingFace repo
// directly; the .safetensors is lazy-downloaded and cached on first use (no re-hosting). The
// runtime applicator (KleinLoRA / KleinLoRASwapper, in the Klein lib) already handles the
// BFL-fused vs diffusers-split dialects, so the registry is pure metadata + resolution.
//
// Unlike a flat weights-only registry, an entry can declare ORDERED, ROLE-TYPED conditioning
// slots (`kind: "ic"`). The first entry — the self-trained RefControl pose LoRA — takes two
// control images whose ORDER is load-bearing (slot 1 = OpenPose skeleton, slot 2 = subject
// reference; swapping them swaps the roles). `orderedInputs(bindings:)` turns a role→file map
// into the slot-ordered input list the edit pipeline expects, enforcing that constraint in data
// so no consumer has to special-case a particular LoRA.

import Foundation

/// One reference-conditioning slot an `ic` adapter declares. Slots are 1-indexed and their ORDER
/// is semantic, so consumers must feed them in `slot` order (see `LoRAEntry.orderedInputs`).
public struct ConditioningSlot: Codable, Sendable, Equatable {
    public let slot: Int
    public let role: String
    /// Attachment media kind. Only `"image"` today.
    public let media: String
    public let required: Bool
    /// Pipeline ingest hint (e.g. `"openpose-skeleton"`, `"raw"`); advisory, may be nil.
    public let ingest: String?
    public let note: String?
}

/// Adapter provenance / gate evidence carried alongside the weights (rank, tensor counts, the
/// dialect the applicator matched, which dtypes it was validated at). Advisory — all optional so
/// a minimal entry still decodes.
public struct AdapterVerification: Codable, Sendable, Equatable {
    public let headerDialect: String?
    public let rank: Int?
    public let alpha: Int?
    public let tensors: Int?
    public let doubleTargets: Int?
    public let singleTargets: Int?
    public let dtypes: [String]?
    public let checkpoint: String?
    public let gate: String?
}

/// One registry adapter. `repo`/`weightFile` resolve to a HF `resolve/main` download URL.
public struct LoRAEntry: Codable, Sendable, Equatable {
    // Identity
    public let id: String
    public let displayName: String
    public let repo: String
    public let weightFile: String
    public let trigger: String
    public let defaultStrength: Float
    // Classification
    /// `"plain"` (weights only) or `"ic"` (needs reference conditioning — see `conditioning`).
    public let kind: String
    /// Capability the adapter targets: `"textToImage"` or `"imageEdit"`.
    public let surface: String
    // Gates / provenance (optional)
    public let status: String?
    public let license: String?
    public let gatedDownload: Bool?
    /// Upstream provenance for external / mirrored adapters (e.g. the CivitAI or origin URL).
    public let source: String?
    /// Default prompt when the caller supplies none (the adapter's canonical usage phrasing).
    public let promptTemplate: String?
    public let conditioning: [ConditioningSlot]?
    public let verification: AdapterVerification?

    /// True for reference-conditioned adapters (declare + consume `conditioning` slots).
    public var isReferenceConditioned: Bool { kind == "ic" }

    /// Declared slots in slot order.
    public var slots: [ConditioningSlot] {
        (conditioning ?? []).sorted { $0.slot < $1.slot }
    }

    /// Required slots in slot order.
    public var requiredSlots: [ConditioningSlot] {
        slots.filter(\.required)
    }

    /// Resolve a role→file map into the slot-ordered input list the pipeline expects. Enforces the
    /// declared order (load-bearing for RefControl), rejects unknown roles, and fails if a required
    /// slot is unbound. Optional slots that are unbound are simply omitted.
    public func orderedInputs(bindings: [String: URL]) throws -> [URL] {
        let known = Set(slots.map(\.role))
        for role in bindings.keys where !known.contains(role) {
            throw LoRARegistryError.unknownSlot(role, adapter: id)
        }
        var ordered: [URL] = []
        for s in slots {
            if let url = bindings[s.role] {
                ordered.append(url)
            } else if s.required {
                throw LoRARegistryError.missingSlot(s.role, adapter: id)
            }
        }
        return ordered
    }
}

/// The decoded registry plus id lookup + surface filtering. Decoding ignores unknown keys, so a
/// future richer manifest still loads here.
public struct LoRARegistry: Codable, Sendable {
    public let schemaVersion: Int
    public let base: String
    public let adapters: [LoRAEntry]

    public func entry(id: String) -> LoRAEntry? { adapters.first { $0.id == id } }

    /// Adapters for a given capability (nil = all).
    public func surfaced(_ surface: String? = nil) -> [LoRAEntry] {
        guard let surface else { return adapters }
        return adapters.filter { $0.surface == surface }
    }

    /// Load the manifest bundled with the package.
    public static func bundled() throws -> LoRARegistry {
        guard let url = Bundle.module.url(forResource: "klein-lora-registry", withExtension: "json")
        else { throw LoRARegistryError.manifestMissing }
        return try JSONDecoder().decode(LoRARegistry.self, from: Data(contentsOf: url))
    }

    /// Load a manifest from an explicit path (e.g. a vendored copy under a source-of-truth repo).
    public static func load(from url: URL) throws -> LoRARegistry {
        try JSONDecoder().decode(LoRARegistry.self, from: Data(contentsOf: url))
    }
}

public enum LoRARegistryError: Error, LocalizedError {
    case manifestMissing
    case unknownAdapter(String)
    case missingSlot(String, adapter: String)
    case unknownSlot(String, adapter: String)
    case download(String, underlying: String)

    public var errorDescription: String? {
        switch self {
        case .manifestMissing:
            return "Bundled Klein LoRA registry manifest not found."
        case .unknownAdapter(let id):
            return "No LoRA with id '\(id)' in the registry."
        case .missingSlot(let role, let adapter):
            return "LoRA '\(adapter)' requires a '\(role)' conditioning input that was not provided."
        case .unknownSlot(let role, let adapter):
            return "LoRA '\(adapter)' has no conditioning slot named '\(role)'."
        case .download(let id, let underlying):
            return "Failed to download LoRA '\(id)': \(underlying)"
        }
    }
}

/// Lazy-downloads + caches registry LoRAs from HuggingFace `resolve/main`, local-store first.
public struct LoRACache: Sendable {
    public let directory: URL

    public init(directory: URL) { self.directory = directory }

    /// Canonical cached path for an entry (id-named so a `weightFile` rename never forks the cache).
    public func localURL(for entry: LoRAEntry) -> URL {
        directory.appendingPathComponent("\(entry.id).safetensors")
    }

    /// Return the local weights, downloading on first use. Resolution order:
    ///   1. the id-named cache file,
    ///   2. a pre-placed file under the entry's own `weightFile` name, or `<id>/<weightFile>` —
    ///      so weights already sitting in the model store are used with no network round-trip,
    ///   3. download from HF `resolve/main` (atomic: temp file, then move).
    public func ensure(_ entry: LoRAEntry) async throws -> URL {
        let dest = localURL(for: entry)
        let fm = FileManager.default
        if fm.fileExists(atPath: dest.path) { return dest }

        // Local-store hits — use in place (no copy, no download).
        let preplaced = [
            directory.appendingPathComponent(entry.weightFile),
            directory.appendingPathComponent(entry.id).appendingPathComponent(entry.weightFile),
        ]
        for p in preplaced where fm.fileExists(atPath: p.path) { return p }

        try fm.createDirectory(at: directory, withIntermediateDirectories: true)

        var comps = URLComponents()
        comps.scheme = "https"
        comps.host = "huggingface.co"
        // Percent-encode the path so spaced / non-ASCII weight names resolve.
        comps.path = "/\(entry.repo)/resolve/main/\(entry.weightFile)"
        guard let url = comps.url else {
            throw LoRARegistryError.download(entry.id, underlying: "bad URL for \(entry.weightFile)")
        }
        do {
            let (tmp, response) = try await URLSession.shared.download(from: url)
            if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                throw LoRARegistryError.download(entry.id, underlying: "HTTP \(http.statusCode)")
            }
            try? fm.removeItem(at: dest)
            try fm.moveItem(at: tmp, to: dest)
            return dest
        } catch is CancellationError {
            // Never launder a cancellation into a registry error — the engine's lane
            // disambiguation and the caller's .cancelled classification key on the type.
            throw CancellationError()
        } catch let e as LoRARegistryError {
            throw e
        } catch {
            throw LoRARegistryError.download(entry.id, underlying: error.localizedDescription)
        }
    }
}

/// `metaData` keys a package reads for per-request LoRA selection (used when the registry is
/// wired through the ModelPackage surface; the CLI drives selection by flag instead).
public enum LoRAMetaKeys {
    /// Registry id of the LoRA to apply (absent / empty = pure base).
    public static let id = "loraId"
    /// Optional strength override; defaults to the entry's `defaultStrength`.
    public static let strength = "loraStrength"
}
