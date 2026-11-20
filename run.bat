@echo off
setlocal
cd /d "%~dp0"
title Entity Tracker Launcher

rem ---- first-time setup: virtual environment + requirements ----
set "PY=python"
where python >nul 2>nul || set "PY=py"

if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    %PY% -m venv venv
    if errorlevel 1 (
        echo.
        echo Python was not found. Install Python 3.10-3.12 from python.org
        echo and tick "Add python.exe to PATH" during setup, then run this again.
        pause
        exit /b 1
    )
)

if not exist "venv\.deps_installed" (
    echo Installing requirements. First time only, this can take several minutes...
    "venv\Scripts\python.exe" -m pip install --upgrade pip
    "venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Install failed. Check your internet connection and try again.
        pause
        exit /b 1
    )
    echo ok> "venv\.deps_installed"
)

:menu
cls
echo ==========================================
echo         ENTITY TRACKER LAUNCHER
echo ==========================================
echo.
echo   1  Collect screenshots (window with buttons)
echo   2  Import CVAT export into the dataset
echo   3  Check dataset (stats)
echo   4  Train the model
echo   5  Test the model on test images
echo   6  RUN the tracker overlay
echo   7  Open the screenshots folder
echo   8  Test screen capture
echo   0  Exit
echo.
set "choice="
set /p "choice=Choose a number: "

if "%choice%"=="1" ( "venv\Scripts\python.exe" collector_gui.py & goto menu )
if "%choice%"=="2" goto import
if "%choice%"=="3" ( "venv\Scripts\python.exe" dataset_tools.py stats & goto done )
if "%choice%"=="4" goto train
if "%choice%"=="5" goto testmodel
if "%choice%"=="6" goto runapp
if "%choice%"=="7" ( if not exist "dataset\raw_images" mkdir "dataset\raw_images" & start "" "dataset\raw_images" & goto menu )
if "%choice%"=="8" ( "venv\Scripts\python.exe" capture.py --seconds 5 & goto done )
if "%choice%"=="0" exit /b 0
goto menu

:import
echo.
echo Drag your CVAT export .zip onto this window and press Enter.
set "zip="
set /p "zip=Path: "
if "%zip%"=="" goto menu
"venv\Scripts\python.exe" dataset_tools.py import-cvat %zip% --clean
goto done

:train
echo.
echo Training on CPU is slow. Press Ctrl+C to stop; close this window to cancel.
"venv\Scripts\python.exe" train.py --check-only
if errorlevel 1 goto done
"venv\Scripts\python.exe" train.py --model yolo11n.pt --imgsz 480 --epochs 60 --workers 0
goto done

:testmodel
if not exist "models\best.pt" (
    echo.
    echo No trained model yet. Use option 4 first.
    goto done
)
"venv\Scripts\python.exe" test_detection.py --images dataset\images\test --out detection_test
if exist "detection_test" start "" "detection_test"
goto done

:runapp
if not exist "models\best.pt" (
    echo.
    echo No trained model yet ^(models\best.pt is missing^). Collect, label and train first: options 1, 2, 4.
    goto done
)
"venv\Scripts\python.exe" main.py
goto done

:done
echo.
pause
goto menu
