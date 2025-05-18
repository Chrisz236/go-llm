package anthropic

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
	defaultAPIEndpoint = "https://api.anthropic.com/v1/messages"
	defaultTimeout     = 30 * time.Second
	defaultAPIVersion  = "2023-06-01"
)

// Provider implements the llm.Provider interface for Anthropic
type Provider struct {
	apiKey     string
	apiVersion string
	endpoint   string
	client     *http.Client
	modelList  []string
}

// NewProvider creates a new Anthropic provider
func NewProvider() *Provider {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	return NewProviderWithKey(apiKey)
}

// NewProviderWithKey creates a new Anthropic provider with the given API key
func NewProviderWithKey(apiKey string) *Provider {
	return &Provider{
		apiKey:     apiKey,
		apiVersion: defaultAPIVersion,
		endpoint:   defaultAPIEndpoint,
		client: &http.Client{
			Timeout: defaultTimeout,
		},
		modelList: []string{
			"claude-3-7-sonnet-20250219",
			"claude-3-opus-20240229",
			"claude-3-sonnet-20240229",
			"claude-3-haiku-20240307",
			"claude-2.1",
			"claude-2.0",
			"claude-instant-1.2",
			// Add more models as needed
		},
	}
}

// Name returns the name of the provider
func (p *Provider) Name() string {
	return "anthropic"
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

// Convert LLM messages to Anthropic format
func convertMessages(messages []llm.Message) ([]anthropicMessage, string) {
	anthropicMessages := []anthropicMessage{}
	system := ""

	for _, msg := range messages {
		if msg.Role == "system" {
			system = msg.Content
		} else {
			role := msg.Role
			if role == "assistant" {
				role = "assistant"
			} else {
				role = "user"
			}
			anthropicMessages = append(anthropicMessages, anthropicMessage{
				Role:    role,
				Content: msg.Content,
			})
		}
	}

	return anthropicMessages, system
}

// anthropicMessage represents an Anthropic message
type anthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// anthropicRequest represents an Anthropic messages API request
type anthropicRequest struct {
	Model         string             `json:"model"`
	Messages      []anthropicMessage `json:"messages"`
	System        string             `json:"system,omitempty"`
	MaxTokens     int                `json:"max_tokens,omitempty"`
	Temperature   float64            `json:"temperature,omitempty"`
	TopP          float64            `json:"top_p,omitempty"`
	Stream        bool               `json:"stream,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
}

// anthropicResponseContent represents content in an Anthropic response
type anthropicResponseContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// anthropicResponse represents an Anthropic messages API response
type anthropicResponse struct {
	ID           string                     `json:"id"`
	Type         string                     `json:"type"`
	Role         string                     `json:"role"`
	Content      []anthropicResponseContent `json:"content"`
	Model        string                     `json:"model"`
	StopReason   string                     `json:"stop_reason"`
	StopSequence string                     `json:"stop_sequence"`
	Usage        anthropicUsage             `json:"usage"`
}

// anthropicUsage represents token usage in an Anthropic response
type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// Completion sends a completion request to the Anthropic API
func (p *Provider) Completion(ctx context.Context, req *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	if p.apiKey == "" {
		return nil, fmt.Errorf("Anthropic API key not set")
	}

	// Convert messages to Anthropic format
	messages, system := convertMessages(req.Messages)

	// Create Anthropic request
	anthropicReq := anthropicRequest{
		Model:    req.Model,
		Messages: messages,
		System:   system,
		Stream:   false,
	}

	// Set optional parameters if provided
	if req.MaxTokens != nil {
		anthropicReq.MaxTokens = *req.MaxTokens
	} else {
		anthropicReq.MaxTokens = 4096 // Default to a reasonable value
	}

	if req.Temperature != nil {
		anthropicReq.Temperature = *req.Temperature
	}

	if req.TopP != nil {
		anthropicReq.TopP = *req.TopP
	}

	if req.Stop != nil {
		anthropicReq.StopSequences = req.Stop
	}

	// Apply extra parameters if provided
	if req.ExtraParams != nil {
		// Add Anthropic-specific parameters as needed
	}

	// Marshal request to JSON
	reqBody, err := json.Marshal(anthropicReq)
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
	httpReq.Header.Set("x-api-key", p.apiKey)
	httpReq.Header.Set("anthropic-version", p.apiVersion)

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
		return nil, fmt.Errorf("Anthropic API returned error: %s - %s", resp.Status, string(body))
	}

	// Parse response
	var anthropicResp anthropicResponse
	if err := json.Unmarshal(body, &anthropicResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Extract text from content
	var content string
	for _, c := range anthropicResp.Content {
		if c.Type == "text" {
			content += c.Text
		}
	}

	// Convert Anthropic response to LLM response
	llmResp := &llm.CompletionResponse{
		ID:          anthropicResp.ID,
		Object:      "chat.completion",
		Created:     time.Now().Unix(),
		Model:       anthropicResp.Model,
		Provider:    p.Name(),
		RawResponse: anthropicResp,
		Usage: llm.CompletionUsage{
			PromptTokens:     anthropicResp.Usage.InputTokens,
			CompletionTokens: anthropicResp.Usage.OutputTokens,
			TotalTokens:      anthropicResp.Usage.InputTokens + anthropicResp.Usage.OutputTokens,
		},
		Choices: []llm.CompletionChoice{
			{
				Index: 0,
				Message: llm.Message{
					Role:    "assistant",
					Content: content,
				},
				FinishReason: anthropicResp.StopReason,
			},
		},
	}

	return llmResp, nil
}

// AnthropicResponseStream implements the llm.ResponseStream interface for Anthropic
type AnthropicResponseStream struct {
	reader         *bufReader
	provider       string
	id             string
	streamFinished bool
}

// bufReader helps process SSE data from Anthropic stream
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

// anthropicEvent represents a single event in the Anthropic SSE stream
type anthropicEvent struct {
	Type         string             `json:"type"`
	Message      *anthropicResponse `json:"message,omitempty"`
	ContentBlock *struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content_block,omitempty"`
	Delta *struct {
		Type       string `json:"type"`
		Text       string `json:"text"`
		StopReason string `json:"stop_reason,omitempty"`
	} `json:"delta,omitempty"`
}

// Recv receives the next chunk from the stream
func (s *AnthropicResponseStream) Recv() (*llm.CompletionResponse, error) {
	if s.streamFinished {
		return nil, io.EOF
	}

	for {
		line, err := s.reader.ReadLine()
		if err != nil {
			return nil, err
		}

		// Skip empty lines
		if len(line) == 0 {
			continue
		}

		// Check for data prefix
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}

		// Extract data part
		data := bytes.TrimPrefix(line, []byte("data: "))

		// Check for stream end
		if string(data) == "[DONE]" {
			s.streamFinished = true
			return nil, io.EOF
		}

		// Parse JSON event
		var event anthropicEvent
		if err := json.Unmarshal(data, &event); err != nil {
			return nil, fmt.Errorf("failed to parse stream event: %w", err)
		}

		// Handle different event types
		if event.Type == "content_block_start" || event.Type == "content_block_delta" {
			var content string

			if event.ContentBlock != nil {
				content = event.ContentBlock.Text
			} else if event.Delta != nil {
				content = event.Delta.Text
				if event.Delta.StopReason != "" {
					s.streamFinished = true
				}
			}

			// Create response
			resp := &llm.CompletionResponse{
				ID:       s.id,
				Object:   "chat.completion.chunk",
				Created:  time.Now().Unix(),
				Provider: s.provider,
				Choices: []llm.CompletionChoice{
					{
						Index: 0,
						Message: llm.Message{
							Role:    "assistant",
							Content: content,
						},
					},
				},
			}

			return resp, nil
		} else if event.Type == "message_start" && event.Message != nil {
			s.id = event.Message.ID
		}
	}
}

// Close closes the stream
func (s *AnthropicResponseStream) Close() error {
	return s.reader.Close()
}

// CompletionStream sends a streaming completion request to the Anthropic API
func (p *Provider) CompletionStream(ctx context.Context, req *llm.CompletionRequest) (llm.ResponseStream, error) {
	if p.apiKey == "" {
		return nil, fmt.Errorf("Anthropic API key not set")
	}

	// Convert messages to Anthropic format
	messages, system := convertMessages(req.Messages)

	// Create Anthropic request
	anthropicReq := anthropicRequest{
		Model:    req.Model,
		Messages: messages,
		System:   system,
		Stream:   true,
	}

	// Set optional parameters if provided
	if req.MaxTokens != nil {
		anthropicReq.MaxTokens = *req.MaxTokens
	} else {
		anthropicReq.MaxTokens = 4096 // Default to a reasonable value
	}

	if req.Temperature != nil {
		anthropicReq.Temperature = *req.Temperature
	}

	if req.TopP != nil {
		anthropicReq.TopP = *req.TopP
	}

	if req.Stop != nil {
		anthropicReq.StopSequences = req.Stop
	}

	// Marshal request to JSON
	reqBody, err := json.Marshal(anthropicReq)
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
	httpReq.Header.Set("x-api-key", p.apiKey)
	httpReq.Header.Set("anthropic-version", p.apiVersion)
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
		return nil, fmt.Errorf("Anthropic API returned error: %s - %s", resp.Status, string(body))
	}

	// Create and return the stream
	return &AnthropicResponseStream{
		reader:   newBufReader(resp.Body),
		provider: p.Name(),
	}, nil
}

// Initialize registers the Anthropic provider with the LLM system
func Initialize() {
	provider := NewProvider()
	llm.RegisterProvider(provider)
}

// init is automatically called when the package is imported
func init() {
	Initialize()
}
