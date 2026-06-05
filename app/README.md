# 本地批处理启动版

这个目录用于本机直接双击启动网页。

使用方法：

1. 双击 `run_webapp.bat`。
2. 等待服务启动。
3. 浏览器会自动打开 `http://127.0.0.1:8000/`。
4. 使用期间不要关闭命令行窗口。

注意：

- 这个版本需要电脑上已有可用的 Python/conda 环境。
- 脚本会优先寻找 `D:\anaconda\envs\yolo\python.exe`，找不到时再尝试用户目录下的 anaconda/miniconda。
- 如果用户没有 Python/conda，请使用 `03_windows_installer_exe` 里的安装包。
