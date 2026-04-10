// ── Provider Capabilities ─────────────────────────────────────────────────────
// Single source of truth for what each provider type can do with media.
// When adding a new provider: register its capabilities here only.
// No other file should branch on provider names for multimodal decisions.

export type AudioFormat = "whisper" | "gemini-inline" | "vertex" | null;
export type ImageFormat = "openai-vision" | "gemini-inline" | "vertex" | "anthropic" | null;

export interface ProviderCapabilities {
  supportsImage: boolean;
  supportsAudio: boolean;
  audioFormat: AudioFormat;
  imageFormat: ImageFormat;
  whisperModel?: string; // only when audioFormat === "whisper"
}

/**
 * Map a resolved provider type (from resolveProviderType) to its multimodal capabilities.
 * Input must be already resolved — never pass raw providerType strings here.
 */
export function getProviderCapabilities(resolvedType: string): ProviderCapabilities {
  switch (resolvedType) {
    case "gemini":
      return {
        supportsImage: true,
        supportsAudio: true,
        audioFormat: "gemini-inline",
        imageFormat: "gemini-inline",
      };

    case "vertexai":
      return {
        supportsImage: true,
        supportsAudio: true,
        audioFormat: "vertex",
        imageFormat: "vertex",
      };

    case "openai":
    case "deepseek":
      return {
        supportsImage: true,
        supportsAudio: true,
        audioFormat: "whisper",
        imageFormat: "openai-vision",
        whisperModel: "whisper-1",
      };

    case "groq":
      return {
        supportsImage: true,
        supportsAudio: true,
        audioFormat: "whisper",
        imageFormat: "openai-vision",
        whisperModel: "whisper-large-v3",
      };

    case "openrouter":
      // OpenRouter routes to various models; vision yes, audio transcription no standard endpoint
      return {
        supportsImage: true,
        supportsAudio: false,
        audioFormat: null,
        imageFormat: "openai-vision",
      };

    case "anthropic":
    case "orbit":
    case "agentrouter":
      return {
        supportsImage: true,
        supportsAudio: false,
        audioFormat: null,
        imageFormat: "anthropic",
      };

    default:
      // Unknown / custom provider — attempt OpenAI-compatible vision, skip audio
      return {
        supportsImage: true,
        supportsAudio: false,
        audioFormat: null,
        imageFormat: "openai-vision",
      };
  }
}
