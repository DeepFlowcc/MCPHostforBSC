import OpenAI, { AzureOpenAI, APIError } from 'openai';
import type { ChatCompletionTool } from 'openai/resources/index.js';
import { ChatCompletionMessageParam } from 'openai/resources/index.js';
import {
  FunctionTool,
  ResponseItem,
  ResponseOutputText,
} from 'openai/resources/responses/responses.js';
import type { MCPClient as MCPClientHTTP } from './client-http.js';
import type { MCPClient as MCPClientSSE } from './client-sse.js';
import type { MCPClient as MCPClientStdio } from './client-stdio.js';
import { isGitHubModels, llm, model } from './config/providers.js';
import { ZodToolType as ZodTool } from './config/types.js';
import { logger } from './helpers/logs.js';

const log = logger('agent');

// Maximum number of messages to keep in the context window
const MAX_MESSAGES = 10;

export class TodoAgent {
  private llm: AzureOpenAI | OpenAI | null = null;
  private model: string = model;
  private toolsByClient: { [name: string]: MCPClientHTTP | MCPClientSSE | MCPClientStdio } = {};

  // Callback functions for tool usage events
  public onToolCall: ((toolName: string, args: string) => void) | null = null;
  public onToolResult: ((toolName: string, result: string) => void) | null = null;
  public onError: ((error: string) => void) | null = null;

  private mcpTools: Array<ZodTool> = [];
  constructor() {
    if (!llm) {
      throw new Error('LLM provider is not initialized');
    }
    this.llm = llm;
  }

  getTools() {
    return this.mcpTools.map((tool) => ({
      name: tool.name,
      description: tool.description,
    }));
  }

  appendTools(mcp: MCPClientHTTP | MCPClientSSE | MCPClientStdio, mcpTools: ZodTool[]) {
    this.mcpTools = [...this.mcpTools, ...mcpTools];
    this.toolsByClient = {
      ...this.toolsByClient,
      ...this.mcpTools.reduce((acc, tool) => {
        acc[tool.name] = mcp;
        return acc;
      }, {} as any),
    };
  }

  async *query(query: string) {
    try {
      if (!!isGitHubModels) {
        yield* this.queryChatCompletion(query);
      } else {
        yield* this.queryResponseAPI(query);
      }
    } catch (error) {
      log.error(`Error during query processing: ${error}`);
      
      let errorMessage: string;
      
      if (error instanceof APIError) {
        // Handle specific OpenAI API errors
        if (error.status === 413) {
          errorMessage = "The request was too large for the model to process. Try a shorter query or clearing some context.";
        } else if (error.status === 429) {
          errorMessage = "Rate limit exceeded. Please try again later.";
        } else {
          errorMessage = `API Error (${error.status}): ${error.message}`;
        }
      } else {
        errorMessage = `An unexpected error occurred: ${error instanceof Error ? error.message : String(error)}`;
      }
      
      // Trigger error callback if it exists
      if (this.onError) {
        this.onError(errorMessage);
      }
      
      // Yield the error message as a response
      yield `Error: ${errorMessage}`;
    }
  }

  private async *queryChatCompletion(query: string) {
    log.warn(`Processing user query using the Chat Completion API`);

    log.info(`User query: ${query}`);
    try {
      const messages: ChatCompletionMessageParam[] = [
        {
          role: 'developer',
          content: `You are a helpful assistant that can use tools to answer questions. Never use markdown, reply with plain text only. 
            You have access to the following tools: ${this.mcpTools
              .map((tool) => `${tool.name}: ${tool.description}`)
              .join(', ')}.`,
        },
        {
          role: 'user',
          content: query,
        },
      ];

      const stopAnimation = log.thinking();

      let response;
      try {
        response = await this.llm!.chat.completions.create({
          model: this.model,
          max_tokens: 800,
          messages,
          tools: this.mcpTools.map(TodoAgent.mcpToolToOpenAiToolChatCompletion),
          parallel_tool_calls: false,
        });
      } catch (error) {
        stopAnimation();
        throw error; // Re-throw to be caught by the outer try-catch
      }

      stopAnimation();

      for await (const chunk of response.choices) {
        const tools = chunk?.message.tool_calls;
        const content = chunk?.message.content;
        if (content) {
          yield log.agent(content);
        }

        if (tools) {
          messages.push(chunk?.message);

          for await (const tool of tools) {
            const toolName: string = tool.function.name;
            const toolArgs: string = tool.function.arguments;
            log.info(`Using tool '${toolName}' with arguments: ${toolArgs}`);
            
            // Trigger the tool call callback if it exists
            if (this.onToolCall) {
              this.onToolCall(toolName, toolArgs);
            }

            const mcpClient = this.toolsByClient[toolName];
            if (!mcpClient) {
              log.warn(`Tool '${toolName}' not found. Skipping...`);
              yield `Tool '${toolName}' not found. Skipping...`;
              continue;
            }

            let result;
            try {
              result = await mcpClient.callTool(toolName, toolArgs);
            } catch (err) {
              log.error(`Error calling tool '${toolName}': ${err}`);
              yield `Error calling tool '${toolName}': ${err instanceof Error ? err.message : String(err)}`;
              continue;
            }
            
            if (result.isError) {
              log.error(`Tool '${toolName}' failed: ${result.error}`);
              yield `Tool '${toolName}' failed: ${result.error}`;
              continue;
            }

            const toolOutput = (result.content as any)[0].text;
            log.success(`Tool '${toolName}' result: ${toolOutput}`);
            
            // Trigger the tool result callback if it exists
            if (this.onToolResult) {
              this.onToolResult(toolName, toolOutput.toString());
            }

            messages.push({
              role: 'tool',
              tool_call_id: tool.id,
              content: toolOutput.toString(),
            });
          }

          // Limit message context if getting too large
          if (messages.length > MAX_MESSAGES) {
            // Keep system message and trim oldest messages
            const systemMessage = messages[0];
            messages.splice(1, messages.length - MAX_MESSAGES); // Keep only the latest messages
            messages.unshift(systemMessage); // Add system message back at the beginning
          }

          try {
            const chat = await this.llm!.chat.completions.create({
              model: this.model,
              max_tokens: 800,
              messages,
              tools: this.mcpTools.map(TodoAgent.mcpToolToOpenAiToolChatCompletion),
            });

            for await (const chunk of chat.choices) {
              const message = chunk?.message.content;
              if (message) {
                yield log.agent(message);
              }
            }
          } catch (error) {
            if (error instanceof APIError && error.status === 413) {
              // Request was too large, try again with fewer messages
              log.warn("Request too large, trimming context and retrying");
              
              // Keep only essential messages: system, user's latest request, and last tool results
              const systemMessage = messages[0];
              const latestUserMessage = messages.find(m => m.role === 'user');
              
              // Start fresh with minimal context
              messages.length = 0;
              messages.push(systemMessage);
              if (latestUserMessage) messages.push(latestUserMessage);
              
              // Try again with reduced context
              const retryChat = await this.llm!.chat.completions.create({
                model: this.model,
                max_tokens: 800,
                messages,
                tools: this.mcpTools.map(TodoAgent.mcpToolToOpenAiToolChatCompletion),
              });
              
              for await (const chunk of retryChat.choices) {
                const message = chunk?.message.content;
                if (message) {
                  yield log.agent(message);
                }
              }
            } else {
              throw error; // Re-throw other errors
            }
          }
        }
      }

      yield '\n';
      log.info('Query completed.');
    } catch (error) {
      log.error(`Error in queryChatCompletion: ${error}`);
      throw error; // Re-throw to be handled by the outer try-catch
    }
  }

  private async *queryResponseAPI(query: string) {
    log.warn(`Processing user query using the Responses API`);

    log.info(`User query: ${query}`);
    try {
      const messages: ResponseItem[] = [];

      const stopAnimation = log.thinking();

      let response;
      try {
        response = await this.llm!.responses.create({
          model: this.model,
          instructions: `You are a helpful assistant that can use tools to answer questions. Never use markdown, reply with plain text only. 
            You have access to the following tools: ${this.mcpTools
              .map((tool) => `${tool.name}: ${tool.description}`)
              .join(', ')}.`,
          input: query,
          tools: this.mcpTools.map(TodoAgent.mcpToolToOpenAiToolResponses),
          parallel_tool_calls: false,
        });
      } catch (error) {
        stopAnimation();
        throw error; // Re-throw to be caught by the outer try-catch
      }

      stopAnimation();

      for (const chunk of response.output) {
        if (chunk.type === 'message') {
          yield log.agent((chunk.content[0] as ResponseOutputText).text);
        }

        if (chunk.type === 'function_call') {
          messages.push(chunk as ResponseItem);
          const toolName: string = chunk.name;
          const toolArgs: string = chunk.arguments;
          log.info(`Using tool '${toolName}' with arguments: ${toolArgs}`);
          
          // Trigger the tool call callback if it exists
          if (this.onToolCall) {
            this.onToolCall(toolName, toolArgs);
          }

          const mcpClient = this.toolsByClient[toolName];
          if (!mcpClient) {
            log.warn(`Tool '${toolName}' not found. Skipping...`);
            yield `Tool '${toolName}' not found. Skipping...`;
            continue;
          }

          let result;
          try {
            result = await mcpClient.callTool(toolName, toolArgs);
          } catch (err) {
            log.error(`Error calling tool '${toolName}': ${err}`);
            yield `Error calling tool '${toolName}': ${err instanceof Error ? err.message : String(err)}`;
            continue;
          }
          
          if (result.isError) {
            log.error(`Tool '${toolName}' failed: ${result.error}`);
            yield `Tool '${toolName}' failed: ${result.error}`;
            continue;
          }

          const toolOutput = (result.content as any)[0].text;
          log.success(`Tool '${toolName}' result: ${toolOutput}`);
          
          // Trigger the tool result callback if it exists
          if (this.onToolResult) {
            this.onToolResult(toolName, toolOutput.toString());
          }
        }

        // Limit message context if getting too large
        if (messages.length > MAX_MESSAGES) {
          messages.splice(0, messages.length - MAX_MESSAGES);
        }

        try {
          const chat = await this.llm!.responses.create({
            model: this.model,
            input: messages,
            previous_response_id: response.id,
            tools: this.mcpTools.map(TodoAgent.mcpToolToOpenAiToolResponses),
          });

          for await (const chunk of chat.output) {
            if (chunk.type === 'message') {
              yield log.agent((chunk.content[0] as ResponseOutputText).text);
            }
          }
        } catch (error) {
          if (error instanceof APIError && error.status === 413) {
            // Request was too large, try again with fewer messages
            log.warn("Request too large, trimming context and retrying");
            
            // Start fresh with minimal context
            messages.length = 0;
            
            // Try again with reduced context
            const retryChat = await this.llm!.responses.create({
              model: this.model,
              input: query, // Just use the original query again
              tools: this.mcpTools.map(TodoAgent.mcpToolToOpenAiToolResponses),
            });
            
            for await (const chunk of retryChat.output) {
              if (chunk.type === 'message') {
                yield log.agent((chunk.content[0] as ResponseOutputText).text);
              }
            }
          } else {
            throw error; // Re-throw other errors
          }
        }
      }

      yield '\n';
      log.info('Query completed.');
    } catch (error) {
      log.error(`Error in queryResponseAPI: ${error}`);
      throw error; // Re-throw to be handled by the outer try-catch
    }
  }

  static zodSchemaToParametersSchema(zodSchema: any): {
    type: string;
    properties: Record<string, any>;
    required: string[];
    additionalProperties: boolean;
  } {
    const properties: Record<string, any> = zodSchema.properties || {};
    // Start with the original required array
    const required: string[] = [...(zodSchema.required || [])];
    const additionalProperties: boolean =
      zodSchema.additionalProperties !== undefined
        ? zodSchema.additionalProperties
        : false;
        
    // Remove default values from properties to avoid OpenAI API errors
    // and at the same time, collect all property keys to ensure
    // they are included in the required array
    const allPropertyKeys = new Set(required);
    
    for (const key in properties) {
      // Add all property keys to the set (this will collect both required and optional ones)
      allPropertyKeys.add(key);
      
      // Remove default values if present
      if (properties[key] && 'default' in properties[key]) {
        delete properties[key].default;
      }
      
      // Fix array properties without 'items'
      if (properties[key] && properties[key].type === 'array' && !properties[key].items) {
        // For 'abi' field which is known to be an array of objects
        if (key === 'abi') {
          properties[key].items = {
            type: 'object',
            additionalProperties: false,
            properties: {
              name: { type: 'string' },
              type: { type: 'string' },
              inputs: {
                type: 'array',
                items: {
                  type: 'object',
                  additionalProperties: false,
                  properties: {
                    name: { type: 'string' },
                    type: { type: 'string' }
                  },
                  required: ['name', 'type']
                }
              },
              outputs: {
                type: 'array',
                items: {
                  type: 'object',
                  additionalProperties: false,
                  properties: {
                    name: { type: 'string' },
                    type: { type: 'string' }
                  },
                  required: ['name', 'type']
                }
              },
              stateMutability: { type: 'string' }
            },
            required: ['name', 'type', 'inputs', 'outputs', 'stateMutability']
          };
        }
        // For 'args' field which is an array
        else if (key === 'args') {
          properties[key].items = {
            type: 'string'
          };
        }
        // For any other array types
        else {
          properties[key].items = {
            type: 'string'
          };
        }
      }
    }
    
    // For OpenAI, make all properties required to avoid validation issues
    const allPropertiesRequired: string[] = Array.from(allPropertyKeys);

    return {
      type: 'object',
      properties,
      required: allPropertiesRequired,
      additionalProperties,
    };
  }

  static mcpToolToOpenAiToolChatCompletion(tool: {
    name: string;
    description?: string;
    inputSchema: any;
  }): ChatCompletionTool {
    // Create a deep copy of the input schema to avoid modifying the original
    const inputSchema = JSON.parse(JSON.stringify(tool.inputSchema));
    
    return {
      type: 'function',
      function: {
        strict: true,
        name: tool.name,
        description: tool.description,
        parameters: {
          ...TodoAgent.zodSchemaToParametersSchema(inputSchema),
        },
      },
    };
  }

  static mcpToolToOpenAiToolResponses(tool: {
    name: string;
    description?: string;
    inputSchema: any;
  }): FunctionTool {
    // Create a deep copy of the input schema to avoid modifying the original
    const inputSchema = JSON.parse(JSON.stringify(tool.inputSchema));
    
    return {
      type: 'function',
      strict: true,
      name: tool.name,
      description: tool.description,
      parameters: {
        ...TodoAgent.zodSchemaToParametersSchema(inputSchema),
      },
    };
  }
}
