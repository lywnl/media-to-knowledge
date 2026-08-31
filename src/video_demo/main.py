from __future__ import annotations

from video_demo.api.app import create_app

DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 7999

app = create_app()


def main() -> None:
    """启动本地 API，使用项目约定的默认监听地址和端口。"""

    import uvicorn

    uvicorn.run(app, host=DEFAULT_API_HOST, port=DEFAULT_API_PORT)


if __name__ == "__main__":
    main()
