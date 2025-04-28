import express from 'express';
import http from 'http';
import { Server as SocketIOServer, Socket } from 'socket.io';
import { TodoAgent } from './agent.js';
import { MCPClient as MCPClientHTTP } from './client-http.js';
import { MCPClient as MCPClientSSE } from './client-sse.js';
import { MCPClient as MCPClientStdio } from './client-stdio.js';
import { MCPConfig, MCPSEEServerConfig, ZodToolType } from './config/types.js';
import { logger } from './helpers/logs.js';
import { join } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

// Get the directory of the current module
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const log = logger('host');

// Function to strip ANSI color codes and formatting from text
function stripAnsiCodes(text: string): string {
  // This regex matches ANSI escape sequences including colors and formatting
  return text.replace(/\x1B\[\d+m|\x1B\[\d+;\d+m|\x1B\[\d+;\d+;\d+m/g, '');
}

export class MCPHost {
  private mcpClients: Array<MCPClientHTTP | MCPClientSSE | MCPClientStdio> = [];
  private servers: { serverName: string; server: MCPSEEServerConfig }[] =
    [] as any;

  private agent: TodoAgent = new TodoAgent();
  private config: MCPConfig | undefined;
  private abortController: AbortController = new AbortController();
  
  // Web server components
  private app: any; // Using any type to avoid TypeScript errors
  private server: http.Server;
  private io: SocketIOServer;
  private port: number = 3000;

  constructor(config?: MCPConfig) {
    this.config = config;
    
    // Initialize Express app
    this.app = express();
    this.server = http.createServer(this.app);
    this.io = new SocketIOServer(this.server);
    
    // Configure Express
    this.app.use(express.static(join(__dirname, '../public')));
    this.app.use(express.json());
    
    process.on('SIGINT', () => {
      this.abortController.abort();
    });
    
    // Setup routes and socket events
    this.setupRoutes();
    this.setupSocketEvents();
    this.setupAgentHooks();
  }
  
  private setupRoutes() {
    this.app.get('/', (req: any, res: any) => {
      res.sendFile(join(__dirname, '../public/index.html'));
    });
    
    this.app.get('/api/tools', (req: any, res: any) => {
      res.json(this.agent.getTools());
    });
  }
  
  private setupAgentHooks() {
    // Set up hooks to receive tool call events
    this.agent.onToolCall = (toolName: string, args: string) => {
      // Emit to all connected clients
      this.io.emit('tool_call', { 
        tool: toolName,
        args: args
      });
    };
    
    this.agent.onToolResult = (toolName: string, result: string) => {
      // Emit to all connected clients
      this.io.emit('tool_result', { 
        tool: toolName,
        result: result
      });
    };
  }
  
  private setupSocketEvents() {
    this.io.on('connection', (socket: Socket) => {
      log.info('A client connected');
      
      socket.on('message', async (message: string) => {
        log.info(`Received message: ${message}`);
        
        // Emit thinking status
        socket.emit('status', { type: 'thinking', message: 'Processing...' });
        
        let responseCounter = 0;
        let lastResponse = '';
        let responses: string[] = [];
        
        // Process the message and stream responses
        for await (const response of this.agent.query(message)) {
          responseCounter++;
          
          // Clean the response text by removing ANSI color codes
          const cleanedResponse = stripAnsiCodes(response);
          
          // Store for later consolidation
          if (cleanedResponse.trim()) {
            responses.push(cleanedResponse);
            lastResponse = cleanedResponse;
            
            // Send incremental update (not marked as final)
            socket.emit('response', { 
              id: responseCounter,
              text: cleanedResponse,
              final: false
            });
          }
        }
        
        // If we collected any responses, send a consolidated final response
        if (responses.length > 0) {
          // Send the final consolidated response
          socket.emit('response', {
            id: responseCounter + 1,
            text: lastResponse, // Send only the last meaningful response
            final: true
          });
        }
        
        // Emit ready status
        socket.emit('status', { type: 'ready', message: 'Ready for next query' });
      });
      
      socket.on('disconnect', () => {
        log.info('A client disconnected');
      });
    });
  }

  async connect() {
    try {
      for (const serverName in this.config?.servers) {
        const server = this.config?.servers[serverName];
        let mcp: MCPClientHTTP | MCPClientSSE | MCPClientStdio;

        if (server.type === 'http') {
          if (!server.url) {
            throw new Error(`HTTP server ${serverName} is missing url`);
          }
          log.info(`Connecting to HTTP server ${serverName} at ${server.url}`);
          mcp = new MCPClientHTTP(serverName, server.url);
        } else if (server.type === 'sse') {
          if (!server.url) {
            throw new Error(`SSE server ${serverName} is missing url`);
          }
          log.info(`Connecting to SSE server ${serverName} at ${server.url}`);
          mcp = new MCPClientSSE(serverName, server.url);
        } else if (server.type === 'stdio') {
          if (!server.scriptPath) {
            throw new Error(`Stdio server ${serverName} is missing scriptPath`);
          }
          log.info(`Connecting to Stdio server ${serverName} using script ${server.scriptPath}`);
          mcp = new MCPClientStdio(serverName, server.scriptPath);
        } else {
          throw new Error(`Unsupported server type: ${server.type}`);
        }

        await mcp.connect();
        this.mcpClients.push(mcp);
        this.servers.push({ serverName, server });

        const mcpTools: ZodToolType[] = (await mcp.getAvailableTools()) || [];
        this.agent.appendTools(mcp, mcpTools);
      }
      log.success('Connected to all MCP servers and loaded tools.');
    } catch (err: any) {
      log.error(
        'Failed to connect to MCP server:',
        err?.cause?.code || err?.message || err
      );
    }
  }

  getAvailableTools() {
    return this.agent.getTools();
  }

  async close() {
    // Close MCP clients
    for (const mcp of this.mcpClients) {
      await mcp.close();
    }
    
    // Close web server
    if (this.server) {
      await new Promise<void>((resolve) => {
        this.server.close(() => resolve());
      });
    }
    
    log.info('Closed all connections.');
  }

  async run() {
    try {
      // Start the web server
      this.server.listen(this.port, () => {
        log.success(`MCP Host Web Interface Started at http://localhost:${this.port}`);
        log.info('Connected to the following servers:', this.servers);
        log.info(
          'Available tools:',
          this.agent.getTools().map((tool) => tool.name)
        );
      });
      
      // Wait for the abort signal
      await new Promise<void>((resolve) => {
        this.abortController.signal.addEventListener('abort', () => resolve());
      });
    } catch (err: any) {
      if (err.name === 'AbortError') {
        log.info('Process aborted. Exiting...');
      } else {
        log.error('Error:', err?.cause?.code || err?.message || err);
      }
    } finally {
      await this.close();
      log.info('MCP Host closed.');
      process.exit(0);
    }
  }
}
