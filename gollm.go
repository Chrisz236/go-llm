package gollm

import (
	"context"

	"github.com/Chrisz236/go-llm/llm"
	_ "github.com/Chrisz236/go-llm/providers" // Import providers for initialization
	"github.com/Chrisz236/go-llm/router"
)

// Completion is a convenience function for sending a completion request
func Completion(ctx context.Context, modelID string, messages []llm.Message, opts ...llm.CompletionOption) (*llm.CompletionResponse, error) {
	return llm.Completion(ctx, modelID, messages, opts...)
}

// CompletionStream is a convenience function for sending a streaming completion request
func CompletionStream(ctx context.Context, modelID string, messages []llm.Message, opts ...llm.CompletionOption) (llm.ResponseStream, error) {
	return llm.CompletionStream(ctx, modelID, messages, opts...)
}

// Message is an alias for llm.Message
type Message = llm.Message

// CompletionResponse is an alias for llm.CompletionResponse
type CompletionResponse = llm.CompletionResponse

// ResponseStream is an alias for llm.ResponseStream
type ResponseStream = llm.ResponseStream

// TaskType is an alias for router.TaskType
type TaskType = router.TaskType

// Common task types
const (
	TaskTypeGeneral            = router.TaskTypeGeneral
	TaskTypeCreative           = router.TaskTypeCreative
	TaskTypeCodeGeneration     = router.TaskTypeCodeGeneration
	TaskTypeCodeExplanation    = router.TaskTypeCodeExplanation
	TaskTypeContentModeration  = router.TaskTypeContentModeration
	TaskTypeTextClassification = router.TaskTypeTextClassification
	TaskTypeSummarization      = router.TaskTypeSummarization
	TaskTypeExtraction         = router.TaskTypeExtraction
)

// WithTemperature is an alias for llm.WithTemperature
func WithTemperature(temp float64) llm.CompletionOption {
	return llm.WithTemperature(temp)
}

// WithMaxTokens is an alias for llm.WithMaxTokens
func WithMaxTokens(tokens int) llm.CompletionOption {
	return llm.WithMaxTokens(tokens)
}

// WithTopP is an alias for llm.WithTopP
func WithTopP(topP float64) llm.CompletionOption {
	return llm.WithTopP(topP)
}

// WithUser is an alias for llm.WithUser
func WithUser(user string) llm.CompletionOption {
	return llm.WithUser(user)
}

// WithExtraParams is an alias for llm.WithExtraParams
func WithExtraParams(params map[string]interface{}) llm.CompletionOption {
	return llm.WithExtraParams(params)
}

// Router is an alias for router.Router
type Router = router.Router

// DefaultRouter returns a router with sensible defaults
func DefaultRouter() *Router {
	return router.DefaultRouter()
}

// NewRouter creates a new router with the given options
func NewRouter(opts ...router.RouterOption) *Router {
	return router.NewRouter(opts...)
}

// RouteCompletion routes a completion request to the best model for the task
func RouteCompletion(ctx context.Context, r *Router, taskType TaskType, messages []Message, opts ...llm.CompletionOption) (*CompletionResponse, error) {
	return r.Route(ctx, taskType, messages, opts...)
}

// RouteCompletionStream routes a streaming completion request to the best model
func RouteCompletionStream(ctx context.Context, r *Router, taskType TaskType, messages []Message, opts ...llm.CompletionOption) (ResponseStream, error) {
	return r.RouteStream(ctx, taskType, messages, opts...)
}
