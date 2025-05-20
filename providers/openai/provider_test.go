package openai

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/Chrisz236/go-llm/llm"
	"github.com/stretchr/testify/assert"
)

type ModelAccessibility struct {
	ModelName    string    `json:"model_name"`
	DateAccessed time.Time `json:"date_accessed"`
	Available    bool      `json:"available"`
	PricePerM    float64   `json:"price_per_m"` // Placeholder for future price tracking
	Response     string    `json:"response,omitempty"`
}

func TestAllModels(t *testing.T) {
	provider := NewProvider()
	if provider.apiKey == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	// Create a slice to store accessibility results
	var accessibilityResults []ModelAccessibility

	// Test each model
	for _, model := range provider.modelList {
		t.Run(model, func(t *testing.T) {
			// Create a simple request
			req := &llm.CompletionRequest{
				Model: model,
				Messages: []llm.Message{
					{
						Role:    "user",
						Content: "Say hello in one word.",
					},
				},
			}

			// Set max tokens based on model type
			maxTokens := 10
			if isCompletionTokenModel(model) {
				req.ExtraParams = map[string]interface{}{
					"max_completion_tokens": maxTokens,
				}
			} else {
				req.MaxTokens = &maxTokens
			}

			// Try to get completion
			resp, err := provider.Completion(context.Background(), req)

			// Record accessibility
			accessibility := ModelAccessibility{
				ModelName:    model,
				DateAccessed: time.Now(),
				Available:    err == nil,
				PricePerM:    0.0, // Placeholder for future price tracking
			}

			// If model is available, verify response and record it
			if err == nil {
				assert.NotNil(t, resp)
				assert.NotEmpty(t, resp.Choices)
				assert.NotEmpty(t, resp.Choices[0].Message.Content)
				accessibility.Response = resp.Choices[0].Message.Content
			} else {
				t.Logf("Model %s not available: %v", model, err)
			}
			accessibilityResults = append(accessibilityResults, accessibility)
		})
	}

	// Save accessibility results to JSON file
	jsonData, err := json.MarshalIndent(accessibilityResults, "", "  ")
	if err != nil {
		t.Fatalf("Failed to marshal accessibility results: %v", err)
	}

	// Create directory if it doesn't exist
	dir := "./"
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatalf("Failed to create directory: %v", err)
	}

	err = os.WriteFile("./model_accessibility.json", jsonData, 0644)
	if err != nil {
		t.Fatalf("Failed to write accessibility results: %v", err)
	}
}

// Helper function to check if a model uses max_completion_tokens
func isCompletionTokenModel(model string) bool {
	completionTokenModels := map[string]bool{
		"o1":                    true,
		"o1-mini":               true,
		"o3-mini":               true,
		"o3-mini-2025-01-31":    true,
		"o4-mini":               true,
		"o4-mini-2025-04-16":    true,
		"o1-mini-2024-09-12":    true,
		"o1-preview":            true,
		"o1-preview-2024-09-12": true,
		"o1-2024-12-17":         true,
	}

	return completionTokenModels[model]
}
