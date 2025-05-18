package providers

import (
	// Import providers for side-effect initialization
	_ "github.com/Chrisz236/go-llm/providers/anthropic"
	_ "github.com/Chrisz236/go-llm/providers/google"
	_ "github.com/Chrisz236/go-llm/providers/openai"
	// Add more providers as they are implemented
)

// Initialize ensures all providers are registered
func Initialize() {
	// This function is a no-op, but importing this package will trigger
	// the init functions of all the provider packages
}
