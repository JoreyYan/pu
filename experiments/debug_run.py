"""调试入口：在运行 `train.main()` 前启动 debugpy，等待 VS Code 附着。

使用方法（在项目根目录运行）：
  python -m experiments.debug_run

然后在 VS Code 中使用 "Python: Attach" 或在 `launch.json` 中配置附着到 5678 端口。
"""
import os
import sys

try:
    import debugpy
except Exception:
    debugpy = None


def wait_for_debugger(host: str = "0.0.0.0", port: int = 5678, wait: bool = True):
    if debugpy is None:
        print("debugpy not installed. Install with: pip install debugpy")
        return
    addr = (host, port)
    print(f"debugpy listening on {addr}. Waiting for VS Code to attach...")
    debugpy.listen(addr)
    if wait:
        debugpy.wait_for_client()
        print("Debugger attached.")


def main():
    # 避免 hydra 改变工作目录时造成相对导入问题，先回到仓库根
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)

    # 启动 debugpy 并等待（默认等待）
    wait_for_debugger()

    # 导入并运行训练脚本的 main
    # train.main() 是由 @hydra.main 装饰的入口，直接调用会使用默认配置
    from experiments import train

    # 可选：在这里设置种子以获得可重复性
    try:
        train.set_global_seed(0)
    except Exception:
        pass

    # 调用主入口（等同于运行 `python experiments/train.py`）
    train.main()


if __name__ == "__main__":
    main()
