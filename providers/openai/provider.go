package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/Chrisz236/go-llm/llm"
)

const (
	defaultAPIEndpoint = "https://api.openai.com/v1/chat/completions"
	defaultTimeout     = 30 * time.Second
)

// Provider implements the llm.Provider interface for OpenAI
type Provider struct {
	apiKey    string
	endpoint  string
	client    *http.Client
	modelList []string
}

// NewProvider creates a new OpenAI provider
func NewProvider() *Provider {
	apiKey := os.Getenv("OPENAI_API_KEY")
	return NewProviderWithKey(apiKey)
}

// NewProviderWithKey creates a new OpenAI provider with the given API key
func NewProviderWithKey(apiKey string) *Provider {
	return &Provider{
		apiKey:   apiKey,
		endpoint: defaultAPIEndpoint,
		client: &http.Client{
			Timeout: defaultTimeout,
		},
		modelList: []string{
			"gpt-3.5-turbo",
			"gpt-3.5-turbo-16k",
			"gpt-4",
			"gpt-4-turbo",
			"gpt-4o",
			// Add more models as needed
		},
	}
}

// Name returns the name of the provider
func (p *Provider) Name() string {
	return "openai"
}

// SupportsModel checks if the provider supports the given model
func (p *Provider) SupportsModel(model string) bool {
	for _, m := range p.modelList {
		if m == model {
			return true
		}
	}
	return false
}

// openAIMessage represents an OpenAI message
type openAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// openAIRequest represents an OpenAI chat completion request
type openAIRequest struct {
	Model            string          `json:"model"`
	Messages         []openAIMessage `json:"messages"`
	Temperature      *float64        `json:"temperature,omitempty"`
	MaxTokens        *int            `json:"max_tokens,omitempty"`
	TopP             *float64        `json:"top_p,omitempty"`
	FrequencyPenalty *float64        `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64        `json:"presence_penalty,omitempty"`
	Stop             []string        `json:"stop,omitempty"`
	Stream           bool            `json:"stream,omitempty"`
	N                int             `json:"n,omitempty"`
	LogitBias        map[string]int  `json:"logit_bias,omitempty"`
	User             string          `json:"user,omitempty"`
}

// openAIResponseChoice represents a choice in an OpenAI response
type openAIResponseChoice struct {
	Index        int           `json:"index"`
	Message      openAIMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

// openAIResponseUsage represents token usage in an OpenAI response
type openAIResponseUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// openAIResponse represents an OpenAI chat completion response
type openAIResponse struct {
	ID                string                 `json:"id"`
	Object            string                 `json:"object"`
	Created           int64                  `json:"created"`
	Model             string                 `json:"model"`
	Choices           []openAIResponseChoice `json:"choices"`
	Usage             openAIResponseUsage    `json:"usage"`
	SystemFingerprint string                 `json:"system_fingerprint,omitempty"`
}

// Completion sends a completion request to the OpenAI API
func (p *Provider) Completion(ctx context.Context, req *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	if p.apiKey == "" {
		return nil, fmt.Errorf("OpenAI API key not set")
	}

	// Convert llm.CompletionRequest to openAIRequest
	openAIReq := openAIRequest{
		Model:            req.Model,
		Temperature:      req.Temperature,
		MaxTokens:        req.MaxTokens,
		TopP:             req.TopP,
		FrequencyPenalty: req.FrequencyPenalty,
		PresencePenalty:  req.PresencePenalty,
		Stop:             req.Stop,
		Stream:           false, // Ensure stream is false for non-streaming requests
		LogitBias:        req.LogitBias,
		User:             req.User,
		N:                1, // Default to 1 completion
	}

	// Convert messages
	openAIReq.Messages = make([]openAIMessage, len(req.Messages))
	for i, msg := range req.Messages {
		openAIReq.Messages[i] = openAIMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	// Apply extra parameters if provided
	if req.ExtraParams != nil {
		if n, ok := req.ExtraParams["n"].(int); ok {
			openAIReq.N = n
		}
		// Add other OpenAI-specific parameters as needed
	}

	// Convert request to JSON
	reqBody, err := json.Marshal(openAIReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.endpoint, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

	// Send request
	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	// Check for error
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("OpenAI API returned error: %s - %s", resp.Status, string(body))
	}

	// Parse response
	var openAIResp openAIResponse
	if err := json.Unmarshal(body, &openAIResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Convert openAIResponse to llm.CompletionResponse
	llmResp := &llm.CompletionResponse{
		ID:                openAIResp.ID,
		Object:            openAIResp.Object,
		Created:           openAIResp.Created,
		Model:             openAIResp.Model,
		SystemFingerprint: openAIResp.SystemFingerprint,
		Provider:          p.Name(),
		RawResponse:       openAIResp,
		Usage: llm.CompletionUsage{
			PromptTokens:     openAIResp.Usage.PromptTokens,
			CompletionTokens: openAIResp.Usage.CompletionTokens,
			TotalTokens:      openAIResp.Usage.TotalTokens,
		},
	}

	// Convert choices
	llmResp.Choices = make([]llm.CompletionChoice, len(openAIResp.Choices))
	for i, choice := range openAIResp.Choices {
		llmResp.Choices[i] = llm.CompletionChoice{
			Index:        choice.Index,
			FinishReason: choice.FinishReason,
			Message: llm.Message{
				Role:    choice.Message.Role,
				Content: choice.Message.Content,
			},
		}
	}

	return llmResp, nil
}

// openAIStreamChunk represents a chunk in a streamed OpenAI response
type openAIStreamChunk struct {
	ID                string               `json:"id"`
	Object            string               `json:"object"`
	Created           int64                `json:"created"`
	Model             string               `json:"model"`
	Choices           []openAIStreamChoice `json:"choices"`
	SystemFingerprint string               `json:"system_fingerprint,omitempty"`
}

// openAIStreamChoice represents a choice in a streamed OpenAI response
type openAIStreamChoice struct {
	Index        int               `json:"index"`
	Delta        openAIStreamDelta `json:"delta"`
	FinishReason string            `json:"finish_reason"`
}

// openAIStreamDelta represents a delta in a streamed OpenAI response
type openAIStreamDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// OpenAIResponseStream implements the llm.ResponseStream interface for OpenAI
type OpenAIResponseStream struct {
	reader         *bufReader
	currentRole    string
	model          string
	provider       string
	id             string
	created        int64
	fingerprint    string
	chunkIndex     int
	streamFinished bool
}

// bufReader helps process SSE data from OpenAI stream
type bufReader struct {
	reader io.ReadCloser
	buf    bytes.Buffer
}

func newBufReader(reader io.ReadCloser) *bufReader {
	return &bufReader{
		reader: reader,
	}
}

func (b *bufReader) ReadLine() ([]byte, error) {
	for {
		line, err := b.buf.ReadBytes('\n')
		if err == nil {
			return bytes.TrimSpace(line), nil
		}

		if err != io.EOF {
			return nil, err
		}

		// Buffer is empty, read more data
		buffer := make([]byte, 1024)
		n, err := b.reader.Read(buffer)
		if err != nil && err != io.EOF {
			return nil, err
		}

		if n == 0 {
			if len(line) > 0 {
				return bytes.TrimSpace(line), nil
			}
			return nil, io.EOF
		}

		b.buf.Write(buffer[:n])
	}
}

func (b *bufReader) Close() error {
	return b.reader.Close()
}

// Recv receives the next chunk from the stream
func (s *OpenAIResponseStream) Recv() (*llm.CompletionResponse, error) {
	if s.streamFinished {
		return nil, io.EOF
	}

	for {
		line, err := s.reader.ReadLine()
		if err != nil {
			return nil, err
		}

		// Skip empty lines or comments
		if len(line) == 0 || bytes.HasPrefix(line, []byte(":")) {
			continue
		}

		// Check for data prefix
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}

		// Extract data part
		data := bytes.TrimPrefix(line, []byte("data: "))

		// Check for stream end
		if bytes.Equal(data, []byte("[DONE]")) {
			s.streamFinished = true
			return nil, io.EOF
		}

		// Parse JSON chunk
		var chunk openAIStreamChunk
		if err := json.Unmarshal(data, &chunk); err != nil {
			return nil, fmt.Errorf("failed to parse stream chunk: %w", err)
		}

		// Update stream state from first chunk if needed
		if s.id == "" {
			s.id = chunk.ID
			s.model = chunk.Model
			s.created = chunk.Created
			s.fingerprint = chunk.SystemFingerprint
		}

		// Process choices
		if len(chunk.Choices) > 0 {
			choice := chunk.Choices[0]

			// Update role if present
			if choice.Delta.Role != "" {
				s.currentRole = choice.Delta.Role
			}

			// Create response
			resp := &llm.CompletionResponse{
				ID:                s.id,
				Object:            "chat.completion.chunk",
				Created:           s.created,
				Model:             s.model,
				SystemFingerprint: s.fingerprint,
				Provider:          s.provider,
				Choices: []llm.CompletionChoice{
					{
						Index:        choice.Index,
						FinishReason: choice.FinishReason,
						Message: llm.Message{
							Role:    s.currentRole,
							Content: choice.Delta.Content,
						},
					},
				},
			}

			s.chunkIndex++

			return resp, nil
		}
	}
}

// Close closes the stream
func (s *OpenAIResponseStream) Close() error {
	return s.reader.Close()
}

// CompletionStream sends a streaming completion request to the OpenAI API
func (p *Provider) CompletionStream(ctx context.Context, req *llm.CompletionRequest) (llm.ResponseStream, error) {
	if p.apiKey == "" {
		return nil, fmt.Errorf("OpenAI API key not set")
	}

	// Convert llm.CompletionRequest to openAIRequest (same as Completion)
	openAIReq := openAIRequest{
		Model:            req.Model,
		Temperature:      req.Temperature,
		MaxTokens:        req.MaxTokens,
		TopP:             req.TopP,
		FrequencyPenalty: req.FrequencyPenalty,
		PresencePenalty:  req.PresencePenalty,
		Stop:             req.Stop,
		Stream:           true, // Ensure stream is true for streaming requests
		LogitBias:        req.LogitBias,
		User:             req.User,
		N:                1, // Default to 1 completion
	}

	// Convert messages
	openAIReq.Messages = make([]openAIMessage, len(req.Messages))
	for i, msg := range req.Messages {
		openAIReq.Messages[i] = openAIMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	// Apply extra parameters if provided
	if req.ExtraParams != nil {
		if n, ok := req.ExtraParams["n"].(int); ok {
			openAIReq.N = n
		}
		// Add other OpenAI-specific parameters as needed
	}

	// Convert request to JSON
	reqBody, err := json.Marshal(openAIReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.endpoint, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	httpReq.Header.Set("Accept", "text/event-stream")

	// Send request
	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}

	// Check for error
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("OpenAI API returned error: %s - %s", resp.Status, string(body))
	}

	// Create and return the stream
	return &OpenAIResponseStream{
		reader:   newBufReader(resp.Body),
		provider: p.Name(),
	}, nil
}

// Initialize registers the OpenAI provider with the LLM system
func Initialize() {
	provider := NewProvider()
	llm.RegisterProvider(provider)
}

// init is automatically called when the package is imported
func init() {
	Initialize()
}
