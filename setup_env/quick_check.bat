@echo off
echo.
echo 🔍 Python环境快速检查
echo.

REM 切换到项目目录
cd /d "C:\Users\feifa\GitHub\P2W"

REM 检查conda环境
if "%CONDA_DEFAULT_ENV%"=="" (
    echo ❌ 未激活conda环境，正在激活...
    call conda activate cuda312
) else (
    echo ✅ 当前conda环境: %CONDA_DEFAULT_ENV%
)

REM 运行CUDA验证（包含环境检查）
echo.
python setup_env\validate_cuda.py

echo.
echo 按任意键退出...
pause >nul
