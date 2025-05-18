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

	// // Example 2: Direct completion with Anthropic
	// anthropicResponse, err := gollm.Completion(ctx, "anthropic/claude-3-haiku", messages,
	// 	gollm.WithTemperature(0.7),
	// 	gollm.WithMaxTokens(100),
	// )
	// if err != nil {
	// 	fmt.Printf("Anthropic error: %v\n", err)
	// } else {
	// 	fmt.Println("Anthropic response:", anthropicResponse.Choices[0].Message.Content)
	// }

	// // Example 3: Using smart routing to select the best model for a task
	// router := gollm.DefaultRouter()

	// routedResponse, err := gollm.RouteCompletion(ctx, router, gollm.TaskTypeCreative, messages,
	// 	gollm.WithTemperature(0.9),
	// )
	// if err != nil {
	// 	fmt.Printf("Router error: %v\n", err)
	// } else {
	// 	fmt.Printf("Routed to: %s\n", routedResponse.Provider)
	// 	fmt.Println("Routed response:", routedResponse.Choices[0].Message.Content)
	// }

	// // Example 4: Stream a response
	// fmt.Println("\nStreaming response from OpenAI:")
	// stream, err := gollm.CompletionStream(ctx, "openai/gpt-3.5-turbo", messages)
	// if err != nil {
	// 	fmt.Printf("Stream error: %v\n", err)
	// 	return
	// }
	// defer stream.Close()

	// for {
	// 	chunk, err := stream.Recv()
	// 	if err != nil {
	// 		break
	// 	}

	// 	// Print each content chunk as it arrives
	// 	if len(chunk.Choices) > 0 {
	// 		fmt.Print(chunk.Choices[0].Message.Content)
	// 	}
	// }
	// fmt.Println()
}
