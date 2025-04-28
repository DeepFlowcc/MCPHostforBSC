import EventEmitter from 'node:events';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { ToolListChangedNotificationSchema } from '@modelcontextprotocol/sdk/types.js';
import { logger } from './helpers/logs.js';

const log = logger('host');

// Get the directory of the current module
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export class MCPClient extends EventEmitter {
  private client: Client;
  private transport: StdioClientTransport;
  private scriptPath: string;

  constructor(serverName: string, scriptPath: string) {
    super();
    this.client = new Client({
      name: 'mcp-client-' + serverName,
      version: '1.0.0',
    });

    this.scriptPath = scriptPath;
    
    // Create a client transport that spawns the script process
    this.transport = new StdioClientTransport({
      command: "node",
      args: [resolve(__dirname, scriptPath)]
    });

    this.client.setNotificationHandler(
      ToolListChangedNotificationSchema,
      () => {
        log.info('Emitting toolListChanged event');
        this.emit('toolListChanged');
      }
    );
  }

  async connect() {
    try {
      await this.client.connect(this.transport);
      log.success(`Connected to stdio MCP server: ${this.scriptPath}`);
      return true;
    } catch (error) {
      log.error(`Failed to initialize stdio MCP client:`, error);
      return false;
    }
  }

  async getAvailableTools() {
    const result = await this.client.listTools();
    return result.tools;
  }

  async callTool(name: string, toolArgs: string) {
    log.info(`Calling tool ${name} with arguments:`, toolArgs);

    return await this.client.callTool({
      name,
      arguments: JSON.parse(toolArgs),
    });
  }

  async close() {
    log.info('Closing transport...');
    await this.transport.close();
  }
} 