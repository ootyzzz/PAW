@echo off
REM P2W项目环境验证快捷方式
REM 调用setup_env文件夹中的环境验证脚本

echo 🔍 启动环境验证...
cd /d "%~dp0"
python setup_env\validate_cuda.py
pause
