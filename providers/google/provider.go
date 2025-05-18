package google

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/Chrisz236/go-llm/llm"
)

const (
	defaultAPIEndpoint = "https://generativelanguage.googleapis.com/v1beta/models"
	defaultTimeout     = 30 * time.Second
)

// Provider implements the llm.Provider interface for Google's Gemini models
type Provider struct {
	apiKey    string
	endpoint  string
	client    *http.Client
	modelList []string
}

// NewProvider creates a new Google provider
func NewProvider() *Provider {
	apiKey := os.Getenv("GEMINI_API_KEY")
	return NewProviderWithKey(apiKey)
}

// NewProviderWithKey creates a new Google provider with the given API key
func NewProviderWithKey(apiKey string) *Provider {
	return &Provider{
		apiKey:   apiKey,
		endpoint: defaultAPIEndpoint,
		client: &http.Client{
			Timeout: defaultTimeout,
		},
		modelList: []string{
			"gemini-1.5-pro",
			"gemini-1.5-flash",
			"gemini-2.0-pro",
			"gemini-2.0-flash",
			// Add more models as needed
		},
	}
}

// Name returns the name of the provider
func (p *Provider) Name() string {
	return "google"
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

// geminiPart represents a part of a Gemini message
type geminiPart struct {
	Text string `json:"text,omitempty"`
	Role string `json:"role,omitempty"`
}

// geminiContent represents a content message for Gemini API
type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

// geminiRequest represents a Google Gemini API request
type geminiRequest struct {
	Contents         []geminiContent `json:"contents"`
	GenerationConfig *struct {
		Temperature     *float64 `json:"temperature,omitempty"`
		MaxOutputTokens *int     `json:"maxOutputTokens,omitempty"`
		TopP            *float64 `json:"topP,omitempty"`
		TopK            *int     `json:"topK,omitempty"`
		StopSequences   []string `json:"stopSequences,omitempty"`
	} `json:"generationConfig,omitempty"`
	SafetySettings []struct {
		Category  string `json:"category"`
		Threshold string `json:"threshold"`
	} `json:"safetySettings,omitempty"`
	Stream bool `json:"stream,omitempty"`
}

// geminiResponsePart represents a single part in a Gemini response
type geminiResponsePart struct {
	Text string `json:"text"`
}

// geminiResponseContent represents content in a Gemini response
type geminiResponseContent struct {
	Role  string               `json:"role"`
	Parts []geminiResponsePart `json:"parts"`
}

// geminiCandidate represents a single candidate in a Gemini response
type geminiCandidate struct {
	Content      geminiResponseContent `json:"content"`
	FinishReason string                `json:"finishReason"`
	Index        int                   `json:"index"`
	// Safety ratings and other fields omitted for brevity
}

// geminiUsage represents token usage in a Gemini response
type geminiUsage struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

// geminiResponse represents a complete response from Gemini API
type geminiResponse struct {
	Candidates     []geminiCandidate `json:"candidates"`
	PromptFeedback interface{}       `json:"promptFeedback,omitempty"`
	Usage          geminiUsage       `json:"usage,omitempty"`
}

// convertMessagesToGeminiFormat converts LLM messages to Gemini format
func convertMessagesToGeminiFormat(messages []llm.Message) []geminiContent {
	var systemMessage string
	var geminiContents []geminiContent

	// First, extract system message if present
	for _, msg := range messages {
		if msg.Role == "system" {
			systemMessage = msg.Content
			break
		}
	}

	// If we have a system message, start with a special user message
	if systemMessage != "" {
		geminiContents = append(geminiContents, geminiContent{
			Role: "user",
			Parts: []geminiPart{
				{Text: systemMessage},
			},
		})
	}

	// Process the rest of the messages
	for _, msg := range messages {
		if msg.Role == "system" {
			continue // Already handled
		}

		role := msg.Role
		// Map standard roles to Gemini's expected roles
		if role == "assistant" {
			role = "model"
		} else if role != "user" {
			role = "user" // Default to user for non-standard roles
		}

		// Add the message
		geminiContents = append(geminiContents, geminiContent{
			Role: role,
			Parts: []geminiPart{
				{Text: msg.Content},
			},
		})
	}

	return geminiContents
}

// Completion sends a completion request to the Google API
func (p *Provider) Completion(ctx context.Context, req *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	if p.apiKey == "" {
		return nil, fmt.Errorf("Google API key not set")
	}

	// Create the url for the specific model
	url := fmt.Sprintf("%s/%s:generateContent?key=%s", p.endpoint, req.Model, p.apiKey)

	// Convert LLM request to Gemini format
	contents := convertMessagesToGeminiFormat(req.Messages)

	// Create the Gemini request
	geminiReq := geminiRequest{
		Contents: contents,
		GenerationConfig: &struct {
			Temperature     *float64 `json:"temperature,omitempty"`
			MaxOutputTokens *int     `json:"maxOutputTokens,omitempty"`
			TopP            *float64 `json:"topP,omitempty"`
			TopK            *int     `json:"topK,omitempty"`
			StopSequences   []string `json:"stopSequences,omitempty"`
		}{
			Temperature:     req.Temperature,
			MaxOutputTokens: req.MaxTokens,
			TopP:            req.TopP,
			StopSequences:   req.Stop,
		},
		Stream: false,
	}

	// Apply extra parameters if provided
	if req.ExtraParams != nil {
		if topK, ok := req.ExtraParams["topK"].(int); ok {
			geminiReq.GenerationConfig.TopK = &topK
		}
		// Add other Gemini-specific parameters as needed
	}

	// Convert request to JSON
	reqBody, err := json.Marshal(geminiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")

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
		return nil, fmt.Errorf("Google API returned error: %s - %s", resp.Status, string(body))
	}

	// Parse response
	var geminiResp geminiResponse
	if err := json.Unmarshal(body, &geminiResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Check if we have any candidates
	if len(geminiResp.Candidates) == 0 {
		return nil, fmt.Errorf("Google API returned no completion candidates")
	}

	// Convert Gemini response to LLM response
	llmResp := &llm.CompletionResponse{
		ID:          fmt.Sprintf("google-%d", time.Now().UnixNano()),
		Object:      "chat.completion",
		Created:     time.Now().Unix(),
		Model:       req.Model,
		Provider:    p.Name(),
		RawResponse: geminiResp,
		Usage: llm.CompletionUsage{
			PromptTokens:     geminiResp.Usage.PromptTokenCount,
			CompletionTokens: geminiResp.Usage.CandidatesTokenCount,
			TotalTokens:      geminiResp.Usage.TotalTokenCount,
		},
	}

	// Convert candidates to choices
	llmResp.Choices = make([]llm.CompletionChoice, len(geminiResp.Candidates))
	for i, candidate := range geminiResp.Candidates {
		// Combine all text parts
		var content string
		for _, part := range candidate.Content.Parts {
			content += part.Text
		}

		llmResp.Choices[i] = llm.CompletionChoice{
			Index:        candidate.Index,
			FinishReason: candidate.FinishReason,
			Message: llm.Message{
				Role:    "assistant",
				Content: content,
			},
		}
	}

	return llmResp, nil
}

// GeminiResponseStream implements the llm.ResponseStream interface for Google
type GeminiResponseStream struct {
	reader         *bufReader
	provider       string
	streamFinished bool
}

// bufReader helps process SSE data from Google stream
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
func (s *GeminiResponseStream) Recv() (*llm.CompletionResponse, error) {
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

		// Parse JSON chunk
		var chunkResp geminiResponse
		if err := json.Unmarshal(data, &chunkResp); err != nil {
			// This could be another type of event
			if strings.Contains(string(data), "finishReason") {
				s.streamFinished = true
				return nil, io.EOF
			}
			continue
		}

		// Check if we have any candidates
		if len(chunkResp.Candidates) == 0 {
			continue
		}

		// Extract content from the first candidate
		candidate := chunkResp.Candidates[0]
		var content string
		for _, part := range candidate.Content.Parts {
			content += part.Text
		}

		// Create response
		resp := &llm.CompletionResponse{
			ID:       fmt.Sprintf("google-%d", time.Now().UnixNano()),
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
					FinishReason: candidate.FinishReason,
				},
			},
		}

		return resp, nil
	}
}

// Close closes the stream
func (s *GeminiResponseStream) Close() error {
	return s.reader.Close()
}

// CompletionStream sends a streaming completion request to the Google API
func (p *Provider) CompletionStream(ctx context.Context, req *llm.CompletionRequest) (llm.ResponseStream, error) {
	if p.apiKey == "" {
		return nil, fmt.Errorf("Google API key not set")
	}

	// Create the url for the specific model
	url := fmt.Sprintf("%s/%s:streamGenerateContent?key=%s", p.endpoint, req.Model, p.apiKey)

	// Convert LLM request to Gemini format
	contents := convertMessagesToGeminiFormat(req.Messages)

	// Create the Gemini request
	geminiReq := geminiRequest{
		Contents: contents,
		GenerationConfig: &struct {
			Temperature     *float64 `json:"temperature,omitempty"`
			MaxOutputTokens *int     `json:"maxOutputTokens,omitempty"`
			TopP            *float64 `json:"topP,omitempty"`
			TopK            *int     `json:"topK,omitempty"`
			StopSequences   []string `json:"stopSequences,omitempty"`
		}{
			Temperature:     req.Temperature,
			MaxOutputTokens: req.MaxTokens,
			TopP:            req.TopP,
			StopSequences:   req.Stop,
		},
		Stream: true,
	}

	// Apply extra parameters if provided
	if req.ExtraParams != nil {
		if topK, ok := req.ExtraParams["topK"].(int); ok {
			geminiReq.GenerationConfig.TopK = &topK
		}
		// Add other Gemini-specific parameters as needed
	}

	// Convert request to JSON
	reqBody, err := json.Marshal(geminiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
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
		return nil, fmt.Errorf("Google API returned error: %s - %s", resp.Status, string(body))
	}

	// Create and return the stream
	return &GeminiResponseStream{
		reader:   newBufReader(resp.Body),
		provider: p.Name(),
	}, nil
}

// Initialize registers the Google provider with the LLM system
func Initialize() {
	provider := NewProvider()
	llm.RegisterProvider(provider)
}

// init is automatically called when the package is imported
func init() {
	Initialize()
}
