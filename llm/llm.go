package llm

import (
	"context"
	"fmt"
	"strings"
	"sync"
)

// registeredProviders holds all registered LLM providers
var (
	registeredProviders = make(map[string]Provider)
	providerMu          sync.RWMutex
)

// RegisterProvider registers a provider with the system
func RegisterProvider(provider Provider) {
	providerMu.Lock()
	defer providerMu.Unlock()
	registeredProviders[provider.Name()] = provider
}

// GetProvider returns a provider by name
func GetProvider(name string) (Provider, bool) {
	providerMu.RLock()
	defer providerMu.RUnlock()
	provider, ok := registeredProviders[name]
	return provider, ok
}

// ListProviders returns a list of all registered providers
func ListProviders() []string {
	providerMu.RLock()
	defer providerMu.RUnlock()
	providers := make([]string, 0, len(registeredProviders))
	for name := range registeredProviders {
		providers = append(providers, name)
	}
	return providers
}

// parseModelIdentifier parses a model identifier in the format "provider/model"
func parseModelIdentifier(modelID string) (provider, model string, err error) {
	parts := strings.SplitN(modelID, "/", 2)
	if len(parts) != 2 {
		return "", "", fmt.Errorf("invalid model identifier: %s, expected format 'provider/model'", modelID)
	}
	return parts[0], parts[1], nil
}

// getProviderForModel returns the appropriate provider for a model
func getProviderForModel(modelID string) (Provider, string, error) {
	providerName, modelName, err := parseModelIdentifier(modelID)
	if err != nil {
		return nil, "", err
	}

	providerMu.RLock()
	defer providerMu.RUnlock()

	provider, ok := registeredProviders[providerName]
	if !ok {
		return nil, "", fmt.Errorf("provider not found: %s", providerName)
	}

	if !provider.SupportsModel(modelName) {
		return nil, "", fmt.Errorf("model %s not supported by provider %s", modelName, providerName)
	}

	return provider, modelName, nil
}

// Completion sends a completion request to the appropriate provider
func Completion(ctx context.Context, modelID string, messages []Message, opts ...CompletionOption) (*CompletionResponse, error) {
	provider, modelName, err := getProviderForModel(modelID)
	if err != nil {
		return nil, err
	}

	req := &CompletionRequest{
		Model:    modelName,
		Messages: messages,
	}

	// Apply options
	for _, opt := range opts {
		opt(req)
	}

	return provider.Completion(ctx, req)
}

// CompletionStream sends a completion request to the appropriate provider and returns a stream
func CompletionStream(ctx context.Context, modelID string, messages []Message, opts ...CompletionOption) (ResponseStream, error) {
	provider, modelName, err := getProviderForModel(modelID)
	if err != nil {
		return nil, err
	}

	req := &CompletionRequest{
		Model:    modelName,
		Messages: messages,
		Stream:   true,
	}

	// Apply options
	for _, opt := range opts {
		opt(req)
	}

	return provider.CompletionStream(ctx, req)
}

// WithTemperature sets the temperature for a completion request
func WithTemperature(temp float64) CompletionOption {
	return func(req *CompletionRequest) {
		req.Temperature = &temp
	}
}

// WithMaxTokens sets the max tokens for a completion request
func WithMaxTokens(tokens int) CompletionOption {
	return func(req *CompletionRequest) {
		req.MaxTokens = &tokens
	}
}

// WithTopP sets the top_p for a completion request
func WithTopP(topP float64) CompletionOption {
	return func(req *CompletionRequest) {
		req.TopP = &topP
	}
}

// WithUser sets the user for a completion request
func WithUser(user string) CompletionOption {
	return func(req *CompletionRequest) {
		req.User = user
	}
}

// WithExtraParams sets additional provider-specific parameters
func WithExtraParams(params map[string]interface{}) CompletionOption {
	return func(req *CompletionRequest) {
		if req.ExtraParams == nil {
			req.ExtraParams = make(map[string]interface{})
		}
		for k, v := range params {
			req.ExtraParams[k] = v
		}
	}
}
