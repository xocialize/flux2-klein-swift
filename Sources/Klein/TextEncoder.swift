// Klein text conditioning — raw Qwen tokenization (no chat template), 3-layer-tap encoder.
// klein feeds the DiT the FULL padded 512-token sequence (verified from diffusers), so this
// pads to `maxSequenceLength` and runs a combined causal + padding-key mask (padded positions
// must match diffusers, which causal-only does NOT — see P4).

import Foundation
import MLX
import Tokenizers

public final class KleinTextEncoder {
    public let encoder: Qwen3HiddenStateEncoder
    public let tokenizer: Tokenizer
    public let maxSequenceLength: Int
    public let padTokenId: Int

    public init(encoder: Qwen3HiddenStateEncoder, tokenizer: Tokenizer,
                maxSequenceLength: Int = 512, padTokenId: Int = 151643) {
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.maxSequenceLength = maxSequenceLength
        self.padTokenId = padTokenId
    }

    /// Qwen chat template (add_generation_prompt=True, enable_thinking=False) — the exact
    /// wrapping diffusers klein feeds the encoder. enable_thinking=False still appends an
    /// EMPTY think block. (Verified against tokenizer.apply_chat_template.)
    public static func formatPrompt(_ prompt: String) -> String {
        "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }

    /// Right-pad to maxSequenceLength; returns (paddedIds, validLen).
    public func tokenize(_ prompt: String) -> (ids: [Int], validLen: Int) {
        var ids = tokenizer.encode(text: Self.formatPrompt(prompt))
        if ids.count > maxSequenceLength { ids = Array(ids.prefix(maxSequenceLength)) }
        let validLen = ids.count
        if ids.count < maxSequenceLength {
            ids += Array(repeating: padTokenId, count: maxSequenceLength - ids.count)
        }
        return (ids, validLen)
    }

    /// Additive mask [1,1,L,L]: -inf where j>i (causal) OR key j is padding (j ≥ validLen).
    static func causalPaddingMask(len: Int, validLen: Int) -> MLXArray {
        var host = [Float](repeating: 0, count: len * len)
        for i in 0..<len {
            for j in 0..<len where j > i || j >= validLen {
                host[i * len + j] = -Float.infinity
            }
        }
        return MLXArray(host, [1, 1, len, len])
    }

    /// prompt → [maxSequenceLength, 7680] concatenated tap features (full padded sequence).
    public func encode(_ prompt: String) -> MLXArray {
        let (ids, validLen) = tokenize(prompt)
        let mask = Self.causalPaddingMask(len: ids.count, validLen: validLen)
        return encoder(MLXArray(ids.map(Int32.init), [1, ids.count]), mask: .array(mask))[0]
    }
}
