import argparse
from random import randint

from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP(
    name="Simple-MCP-Server",
    host="0.0.0.0",  # only used for SSE transport (localhost)
    port=8050,  # only used for SSE transport (set this to any port)
)

# Add a simple calculator tool
@mcp.tool()
def multiply(a: int, b: int) -> int:
    """multipy two numbers together"""
    return a * b


# Add a simple calculator tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

# Add a complicated calculator tool
@mcp.tool()
def metaquasifractal_conjugate_tensorial_operator(a: int) -> int:
    """The Metaquasifractal Conjugate Tensorial Operator, or 𝓜𝒬𝓒𝒯, represents a transformation acting on a quasifractal field φ, modulating its behavior over a pseudo-Riemannian tensor space. It might be defined recursively, with chaotic boundary conditions and deep ties to spectral theory or higher-order symmetry groups."""
    return a**randint(1, a)


# Run the server
if __name__ == "__main__":

    print("Running server with SSE transport")
    mcp.run(transport="sse")
    