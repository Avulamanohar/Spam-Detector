@echo off
setlocal enabledelayedexpansion

echo 🔄 Checking and freeing required ports...

REM === Ports in use ===
set JAVA_PORT=8080
set FASTAPI_PORT=8000
set REACT_PORT=3000

REM Kill processes using these ports
call :kill_port %JAVA_PORT%
call :kill_port %FASTAPI_PORT%
call :kill_port %REACT_PORT%

echo Ports are free. Starting compilation...
echo.

REM === Compile Java Backend (both files) ===
cd /d "D:\sms spam"
echo 🛠 Compiling Java backend files...
javac -cp ".;Data_Base;Data_Base\postgresql-42.7.3.jar;Data_Base\json.jar" Backend\SpamCollectorServer.java Data_Base\SpamDatabase.java
if %errorlevel% neq 0 (
    echo Compilation failed! Check for errors.
    pause
    exit /b
) else (
    echo Compilation successful!
)

echo Starting all services...
echo.

REM === Start Java Backend in background ===
start cmd /k "cd /d D:\sms spam\Backend && java -cp .;..\Data_Base;..\Data_Base\postgresql-42.7.3.jar;..\Data_Base\json.jar SpamCollectorServer"
echo Java Backend started...

REM === Start FastAPI in background ===
start cmd /k "cd /d D:\sms spam\Backend && uvicorn project_FastApi:app --reload --port %FASTAPI_PORT%"
echo FastAPI Server started...

REM === Start React Frontend in background ===
start cmd /k "cd /d D:\sms spam\Frontend\spam-detector-ui && npm start"
echo React Frontend started...

echo.
echo All services are now running! Press any key to close this window (services will keep running)...
pause
exit

:kill_port
REM === Find and kill process using a given port ===
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%1') do (
    echo  Port %1 is in use by PID %%a, killing...
    taskkill /PID %%a /F >nul 2>&1
)
exit /b
