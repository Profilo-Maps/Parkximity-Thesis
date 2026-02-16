@echo off
echo ============================================================
echo ParkXimity Environment Setup
echo ============================================================
echo.
echo This will create a new conda environment with all dependencies.
echo Environment name: parkximity
echo.
echo After setup, GPU support can be added automatically if you
echo have an NVIDIA GPU.
echo.
echo Approximately 3-5 minutes installation time
echo.
pause

echo.
echo Creating conda environment...
conda env create -f environment.yml

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Environment creation failed!
    echo.
    echo Possible solutions:
    echo   1. Update conda: conda update -n base conda
    echo   2. Check internet connection
    echo   3. See SETUP.md for manual installation
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Environment created successfully!
echo ============================================================
echo.
echo To activate the environment, run:
echo   conda activate parkximity
echo.
echo To add GPU support (if you have NVIDIA GPU):
echo   python install_gpu_support.py
echo.
echo To run the analysis:
echo   python Notebooks/Karna/ParkximityCalc.py
echo.
echo See SETUP.md for more information.
echo ============================================================
echo.
echo Launching ParkXimity Configuration UI...
echo.
python Notebooks\Publication\ParkximityConfigUI.py
