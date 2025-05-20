package main

import (
	"context"
	"fmt"
	"os"
	"time"

	gollm "github.com/Chrisz236/go-llm"
	"github.com/Chrisz236/go-llm/router"
)

func main() {
	// Set API keys (in a real application, these should be set in environment variables)
	os.Setenv("OPENAI_API_KEY", "your-openai-key")
	os.Setenv("ANTHROPIC_API_KEY", "your-anthropic-key")

	// Create a context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Create a custom router
	customRouter := gollm.NewRouter(
		router.WithRoutes([]router.ModelRoute{
			// For code generation, prefer GPT-4o
			{TaskType: gollm.TaskTypeCodeGeneration, ModelID: "openai/gpt-4o", Priority: 3, MaxTokens: 8192},
			{TaskType: gollm.TaskTypeCodeGeneration, ModelID: "anthropic/claude-3-opus", Priority: 2, MaxTokens: 200000},

			// For creative writing, prefer Claude
			{TaskType: gollm.TaskTypeCreative, ModelID: "anthropic/claude-3-opus", Priority: 3, MaxTokens: 200000},
			{TaskType: gollm.TaskTypeCreative, ModelID: "openai/gpt-4o", Priority: 2, MaxTokens: 8192},

			// For general tasks, use a variety of models with different priorities
			{TaskType: gollm.TaskTypeGeneral, ModelID: "openai/gpt-3.5-turbo", Priority: 1, MaxTokens: 4096},
			{TaskType: gollm.TaskTypeGeneral, ModelID: "anthropic/claude-3-haiku", Priority: 2, MaxTokens: 200000},
		}),
		router.WithFallbackModel("openai/gpt-3.5-turbo"),
	)

	fmt.Println("=== Code Generation Example ===")
	codeGenMessages := []gollm.Message{
		{Role: "user", Content: "Write a simple Go function that calculates the Fibonacci sequence up to n"},
	}

	codeResponse, err := gollm.RouteCompletion(ctx, customRouter, gollm.TaskTypeCodeGeneration, codeGenMessages)
	if err != nil {
		fmt.Printf("Code generation error: %v\n", err)
	} else {
		fmt.Printf("Routed to: %s\n", codeResponse.Provider)
		fmt.Println(codeResponse.Choices[0].Message.Content)
	}

	fmt.Println("\n=== Creative Writing Example ===")
	creativeMessages := []gollm.Message{
		{Role: "user", Content: "Write a short poem about artificial intelligence"},
	}

	creativeResponse, err := gollm.RouteCompletion(ctx, customRouter, gollm.TaskTypeCreative, creativeMessages)
	if err != nil {
		fmt.Printf("Creative writing error: %v\n", err)
	} else {
		fmt.Printf("Routed to: %s\n", creativeResponse.Provider)
		fmt.Println(creativeResponse.Choices[0].Message.Content)
	}

	fmt.Println("\n=== General Query Example ===")
	generalMessages := []gollm.Message{
		{Role: "user", Content: "What are three interesting facts about dolphins?"},
	}

	generalResponse, err := gollm.RouteCompletion(ctx, customRouter, gollm.TaskTypeGeneral, generalMessages)
	if err != nil {
		fmt.Printf("General query error: %v\n", err)
	} else {
		fmt.Printf("Routed to: %s\n", generalResponse.Provider)
		fmt.Println(generalResponse.Choices[0].Message.Content)
	}

	// Example with streaming
	fmt.Println("\n=== Streaming Example with Smart Routing ===")
	streamMessages := []gollm.Message{
		{Role: "user", Content: "Explain quantum computing in simple terms"},
	}

	stream, err := gollm.RouteCompletionStream(ctx, customRouter, gollm.TaskTypeGeneral, streamMessages)
	if err != nil {
		fmt.Printf("Stream error: %v\n", err)
		return
	}
	defer stream.Close()

	fmt.Println("Streaming response:")
	for {
		chunk, err := stream.Recv()
		if err != nil {
			break
		}

		// Print each content chunk as it arrives
		if len(chunk.Choices) > 0 {
			fmt.Print(chunk.Choices[0].Message.Content)
		}
	}
	fmt.Println()
}
