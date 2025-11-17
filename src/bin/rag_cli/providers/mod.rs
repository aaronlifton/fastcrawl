use anyhow::Result;

mod anthropic;
mod openai;

pub use anthropic::AnthropicProvider;
pub use openai::OpenAiProvider;

/// Trait implemented by concrete LLM providers.
pub trait LlmProvider {
    fn answer(&self, request: &ProviderRequest) -> Result<String>;
}

/// Request envelope shared by the various providers.
pub struct ProviderRequest<'a> {
    pub prompt: &'a str,
    pub temperature: f32,
    pub max_tokens: usize,
}
