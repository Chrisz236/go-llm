# Go-LLM

A universal LLM interface package for Golang that supports multiple vendors (OpenAI, Anthropic, Google, etc.) with a unified API.

## Features

- 🔄 Unified interface for multiple LLM providers
- 🔌 OpenAI-compatible API format as the standard
- 🧩 Modular design for easy extension
- 🔧 Simple, elegant API design

## Installation

```bash
go get github.com/Chrisz236/go-llm
```

## Quick Start

```go
package main

import (
	"context"
	"fmt"
	"os"

	gollm "github.com/Chrisz236/go-llm"
)

func main() {
	// Set API keys (in a real application, these should be set in environment variables)
	os.Setenv("OPENAI_API_KEY", "your-openai-key")
	os.Setenv("ANTHROPIC_API_KEY", "your-anthropic-key")
	os.Setenv("GEMINI_API_KEY", "your-gemini-api-key")

	// Create a context
	ctx := context.Background()

	// Create message for the LLMs
	messages := []gollm.Message{
		{Role: "user", Content: "Hello, how are you?"},
	}

	// OpenAI call
	openaiResponse, err := gollm.Completion(ctx, "openai/gpt-4o", messages,
		gollm.WithTemperature(0.7),
		gollm.WithMaxTokens(100),
	)
	if err != nil {
		fmt.Printf("OpenAI error: %v\n", err)
	} else {
		fmt.Println("OpenAI response:", openaiResponse.Choices[0].Message.Content)
	}

	// Anthropic call
	anthropicResponse, err := gollm.Completion(ctx, "anthropic/claude-3-7-sonnet-20250219", messages,
		gollm.WithTemperature(0.7),
		gollm.WithMaxTokens(100),
	)
	if err != nil {
		fmt.Printf("Anthropic error: %v\n", err)
	} else {
		fmt.Println("Anthropic response:", anthropicResponse.Choices[0].Message.Content)
	}
	
	// Google Gemini call
	googleResponse, err := gollm.Completion(ctx, "google/gemini-2.0-flash", messages,
		gollm.WithTemperature(0.7),
		gollm.WithMaxTokens(100),
	)
	if err != nil {
		fmt.Printf("Google Gemini error: %v\n", err)
	} else {
		fmt.Println("Google Gemini response:", googleResponse.Choices[0].Message.Content)
	}
}
```

## Supported Providers

Currently, Go-LLM supports:

- OpenAI (GPT-3.5, GPT-4, GPT-4o, etc.)
- Anthropic (Claude-3-Opus, Claude-3-Sonnet, Claude-3-Haiku, etc.)
- Google Gemini (Gemini-1.5-Pro, Gemini-1.5-Flash, Gemini-2.0-Pro, Gemini-2.0-Flash)

### OpenAI Models (Tested, ChatCompletion)

| Model Name | Description | Notes |
|------------|-------------|-------|
| chatgpt-4o-latest | Same model used in ChatGPT | |
| gpt-4 | GPT-4 base model | |
| gpt-4.1 | GPT-4.1 base model | |
| gpt-4.1-mini | GPT-4.1 mini variant | |
| gpt-4.1-nano | GPT-4.1 nano variant | |
| gpt-4o | GPT-4 optimized model | |
| gpt-4.5-preview | GPT-4.5 preview | |
| gpt-4o-mini | GPT-4o mini variant | |
| o1 | O1 base model | Uses max_completion_tokens |
| o1-mini | O1 mini variant | Uses max_completion_tokens |
| o3-mini | O3 mini variant | Uses max_completion_tokens |
| o4-mini | O4 mini variant | Uses max_completion_tokens |
| gpt-4-turbo | GPT-4 Turbo | |
| gpt-4-turbo-preview | GPT-4 Turbo preview | |
| gpt-3.5-turbo | GPT-3.5 Turbo | |
| gpt-3.5-turbo-16k | GPT-3.5 Turbo with 16k context | |

Note: All models with date suffixes (e.g., -2024-04-16) are also supported. Models marked with "Uses max_completion_tokens" require the `max_completion_tokens` parameter instead of `max_tokens`.

Feel free to reference the `providers` to add more providers.

## Configuration Options

Go-LLM provides a range of options for configuration:

```go
response, err := gollm.Completion(
    ctx, 
    "openai/gpt-4o", 
    messages,
    gollm.WithTemperature(0.7),
    gollm.WithMaxTokens(2000),
    gollm.WithTopP(0.9),
    gollm.WithUser("user-123"),
)
```

## Architecture

Go-LLM is designed with a modular architecture:

```bash
go-llm/
├── llm/              # Core interfaces and types
├── providers/        # Provider implementations
│   ├── openai/       # OpenAI provider
│   ├── anthropic/    # Anthropic provider
│   ├── google/       # Google Gemini provider
│   └── ...           # Other providers
├── router/           # Smart routing capabilities (Future)
└── examples/         # Usage examples
```

## Future Roadmap

- Smart routing to automatically select the best model for different task types
- Enhanced embedding capabilities
- Function calling support
- Improvements to streaming implementation
- More examples and documentation

## Contributing

Contributions are welcome! Areas to contribute:

- Adding new provider implementations
- Improving documentation
- Adding tests
- Adding new examples

## License

MIT
