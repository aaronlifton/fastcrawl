use std::time::Duration;

use anyhow::{bail, Context, Result};
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};

use super::{LlmProvider, ProviderRequest};

pub struct OpenAiProvider {
    api_key: String,
    model: String,
    client: Client,
}

impl OpenAiProvider {
    pub fn new(api_key: String, model: String) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .context("failed to build OpenAI HTTP client")?;
        Ok(Self {
            api_key,
            model,
            client,
        })
    }
}

impl LlmProvider for OpenAiProvider {
    fn answer(&self, request: &ProviderRequest) -> Result<String> {
        let mut headers = HeaderMap::new();
        let auth = format!("Bearer {}", self.api_key.trim());
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&auth).context("invalid OpenAI API key")?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let body = ChatRequest {
            model: &self.model,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            messages: vec![
                ChatMessage {
                    role: "system",
                    content:
                        "You answer user questions using the provided context chunks. Cite sources as [^chunk_id] and never invent references.",
                },
                ChatMessage {
                    role: "user",
                    content: request.prompt,
                },
            ],
        };
        let resp = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .headers(headers)
            .json(&body)
            .send()
            .context("failed to call OpenAI chat completions")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp
                .text()
                .unwrap_or_else(|_| "<body unavailable>".to_string());
            bail!("OpenAI returned {}: {}", status, text);
        }
        let parsed: ChatResponse = resp.json().context("failed to parse OpenAI response")?;
        let answer = parsed
            .choices
            .into_iter()
            .find_map(|choice| Some(choice.message.content.to_string()))
            .unwrap_or_default();
        Ok(answer)
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    temperature: f32,
    max_tokens: usize,
    messages: Vec<ChatMessage<'a>>,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: AssistantMessage,
}

#[derive(Debug, Deserialize)]
struct AssistantMessage {
    content: String,
}
