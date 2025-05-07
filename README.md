### 1st Prize at DeepFlow AI Hackathon

### MCP Components

The current implementation consists of following main components:

1. **MCP Host**: The main application that interacts with the MCP server and the LLM provider. The host instanciates an LLM provider and provides a terminal interface for the user to interact with the agent.
2. **MCP Client**: The client that communicates with the MCP server using the MCP protocol. The application providers two MCP clients for both HTTP and SSE (Server-Sent Events) protocols.
3. **MCP Server**: The server that implements the MCP protocol and communicates with the Postgres database. The application provides two MCP server implementations: one using HTTP and the other using SSE (Server-Sent Events).
4. **LLM Provider**: The language model provider (e.g., OpenAI, Azure OpenAI, GitHub Models) that generates responses based on the input from the MCP host.
