@echo off
echo.
echo 🚀 P2W项目 - CUDA环境启动
echo.

REM 切换到项目目录
cd /d "C:\Users\feifa\GitHub\P2W"

REM 激活conda环境
echo 📦 激活cuda312环境...
call conda deactivate 2>nul
call conda activate cuda312

REM 验证环境
echo.
echo 🔍 验证CUDA环境...
python setup_env\validate_cuda.py

echo.
echo 🎉 环境验证完成！
echo.

REM 保持窗口打开
cmd /k
