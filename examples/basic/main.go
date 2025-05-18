package main

import (
	"context"
	"fmt"

	gollm "github.com/Chrisz236/go-llm"
)

func main() {
	// Set API keys (in a real application, these should be set in environment variables)
	// os.Setenv("OPENAI_API_KEY", "your-openai-key")
	// os.Setenv("ANTHROPIC_API_KEY", "your-anthropic-key")
	// os.Setenv("GEMINI_API_KEY", "your-google-api-key")

	// Create a context
	ctx := context.Background()

	// Create message for the LLMs
	messages := []gollm.Message{
		{Role: "user", Content: "Hello, how are you?"},
	}

	// Example 1: Direct completion with OpenAI
	openaiResponse, err := gollm.Completion(ctx, "openai/gpt-3.5-turbo", messages,
		gollm.WithTemperature(0.7),
		gollm.WithMaxTokens(100),
	)
	if err != nil {
		fmt.Printf("OpenAI error: %v\n", err)
	} else {
		fmt.Println("OpenAI response:", openaiResponse.Choices[0].Message.Content)
	}

	// Example 2: Direct completion with Anthropic
	anthropicResponse, err := gollm.Completion(ctx, "anthropic/claude-3-7-sonnet-20250219", messages,
		gollm.WithTemperature(0.7),
		gollm.WithMaxTokens(100),
	)
	if err != nil {
		fmt.Printf("Anthropic error: %v\n", err)
	} else {
		fmt.Println("Anthropic response:", anthropicResponse.Choices[0].Message.Content)
	}

	// Example 3: Direct completion with Google Gemini
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
