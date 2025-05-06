import asyncio

import aiohttp
from dotenv import dotenv_values, load_dotenv

load_dotenv()

env = dotenv_values()

chainlit_host = env.get("CHAINLIT_HOST")
chainlit_port = env.get("CHAINLIT_PORT")
mcp_client_entry = env.get("MCP_CLIENT_ENTRY")

assert mcp_client_entry is not None
assert chainlit_host is not None
assert chainlit_port is not None

chainlit_url = f"http://{chainlit_host}:{chainlit_port}"

chainlit_args = [
    "run",
    "-m",
    "chainlit",
    "run",
    "-h",
    "-w",
    mcp_client_entry,
    "--host",
    chainlit_host,
    "--port",
    chainlit_port,
]


async def run_chainlit():
    process = await asyncio.create_subprocess_exec(
        "uv",
        *chainlit_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    asyncio.create_task(read_output(process.stdout, "Chainlit"))

    return process


def print_format_output(name, content):
    print(f"[{name}]: {content}", flush=True)


async def read_output(stream, name):
    while True:
        line = await stream.readline()
        if not line:
            break
        print_format_output(name, line.strip().decode("utf-8"))


async def wait_until_server_ready(url, timeout=5):
    start = asyncio.get_event_loop().time()
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200 or resp.status == 404:
                        return True
        except Exception:
            pass

        await asyncio.sleep(0.2)

        if asyncio.get_event_loop().time() - start > timeout:
            raise RuntimeError("Server not responding")


async def main():
    process = await run_chainlit()
    await wait_until_server_ready(chainlit_url)

    await process.wait()


if __name__ == "__main__":
    asyncio.run(main())
