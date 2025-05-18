package llm

import (
	"context"
	"time"
)

// Message represents a message in a conversation
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// CompletionRequest represents a request to an LLM model
type CompletionRequest struct {
	Model            string                 `json:"model"`
	Messages         []Message              `json:"messages"`
	Temperature      *float64               `json:"temperature,omitempty"`
	MaxTokens        *int                   `json:"max_tokens,omitempty"`
	TopP             *float64               `json:"top_p,omitempty"`
	FrequencyPenalty *float64               `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64               `json:"presence_penalty,omitempty"`
	Stop             []string               `json:"stop,omitempty"`
	Stream           bool                   `json:"stream,omitempty"`
	LogitBias        map[string]int         `json:"logit_bias,omitempty"`
	User             string                 `json:"user,omitempty"`
	ExtraParams      map[string]interface{} `json:"-"` // Provider-specific parameters
}

// CompletionChoice represents a choice in a completion response
type CompletionChoice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// CompletionUsage represents token usage in a completion response
type CompletionUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// CompletionResponse represents a response from an LLM model
type CompletionResponse struct {
	ID                string             `json:"id"`
	Object            string             `json:"object"`
	Created           int64              `json:"created"`
	Model             string             `json:"model"`
	Choices           []CompletionChoice `json:"choices"`
	Usage             CompletionUsage    `json:"usage"`
	SystemFingerprint string             `json:"system_fingerprint,omitempty"`
	Provider          string             `json:"provider"` // Added field to track the provider
	RawResponse       interface{}        `json:"-"`        // The raw response from the provider
}

// CompletionOption defines a function to modify a CompletionRequest
type CompletionOption func(*CompletionRequest)

// Provider defines the interface that all LLM providers must implement
type Provider interface {
	Completion(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error)
	CompletionStream(ctx context.Context, req *CompletionRequest) (ResponseStream, error)
	Name() string
	SupportsModel(model string) bool
}

// ResponseStream defines the interface for streaming responses
type ResponseStream interface {
	Recv() (*CompletionResponse, error)
	Close() error
}

// ModelInfo contains information about a model
type ModelInfo struct {
	ID           string    `json:"id"`
	Name         string    `json:"name"`
	Provider     string    `json:"provider"`
	Capabilities []string  `json:"capabilities"`
	MaxTokens    int       `json:"max_tokens"`
	Created      time.Time `json:"created"`
}
