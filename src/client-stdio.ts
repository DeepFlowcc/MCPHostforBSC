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
    try {
      log.info('Closing stdio transport...');
      
      // Set a timeout to force close if it takes too long
      const forceCloseTimeout = setTimeout(() => {
        log.warn('Forced closing of stdio transport due to timeout');
        // Force kill the child process if it exists
        this.forceKillChildProcess();
      }, 3000); // 3 seconds timeout
      
      await this.transport.close();
      
      // Clear the timeout if normal close succeeded
      clearTimeout(forceCloseTimeout);
      
      log.success('Successfully closed stdio transport');
    } catch (error) {
      log.error('Error closing stdio transport:', error);
      
      // Attempt to force kill the child process as a last resort
      this.forceKillChildProcess();
    }
  }

  /**
   * Attempts to force kill the child process if it exists
   * Uses any to bypass TypeScript checks since the property is not in the type definitions
   */
  private forceKillChildProcess() {
    try {
      // Use any type to access potential internal properties
      const transportAny = this.transport as any;
      
      if (transportAny.childProcess && typeof transportAny.childProcess.kill === 'function') {
        transportAny.childProcess.kill('SIGKILL');
        log.warn('Forced kill of child process');
      }
    } catch (e) {
      log.error('Error force killing child process:', e);
    }
  }
} 