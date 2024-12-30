@echo off
REM 設定變數
set SOURCE_PATH=D:\Git\robotlearning\yolo_inference
set ENV_PATH=D:\Git\robotlearning\yolo_train\env
set OUTPUT_PATH=D:\Git\robotlearning\build_exe

REM 清除舊檔案
rd /s /q "%OUTPUT_PATH%"
mkdir "%OUTPUT_PATH%"

REM 執行 PyInstaller 打包
pyinstaller --noconfirm --onefile --console ^
--add-data "%SOURCE_PATH%\MVS_camera_control.py;." ^
--add-data "%SOURCE_PATH%\Runtime;Runtime/" ^
--add-data "%SOURCE_PATH%\MvImport;MvImport/" ^
--add-data "%SOURCE_PATH%\core;core/" ^
--add-data "%ENV_PATH%\Lib\site-packages\pyfiglet;pyfiglet/" ^
--add-data "%ENV_PATH%\Library\ssl\cacert.pem;ssl/" ^
--add-data "%ENV_PATH%\Library\bin;bin/" ^
--hidden-import torch ^
--hidden-import torchvision ^
--hidden-import cv2 ^
--hidden-import scipy ^
--hidden-import numpy ^
--hidden-import torch.nn.functional ^
--distpath "%OUTPUT_PATH%" ^
--workpath "%OUTPUT_PATH%\build" ^
--specpath "%OUTPUT_PATH%\specs" ^
"%SOURCE_PATH%\main.py"

REM 打包完成後顯示訊息
echo 打包完成，輸出目錄: %OUTPUT_PATH%
pause
