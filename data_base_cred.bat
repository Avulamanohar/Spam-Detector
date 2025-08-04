@echo off
setlocal enabledelayedexpansion

REM === Load DB config from data_base_config.txt ===
for /f "tokens=1,2 delims==" %%A in (data_base_config.txt) do (
    set %%A=%%B
    REM ✅ Permanently save for future sessions
    setx %%A "%%B" >nul
)

REM ✅ Combine DB_HOST_URL + DB_NAME → DB_URL for THIS session
set "DB_URL=%DB_HOST_URL%%DB_NAME%"

REM ✅ Persist DB_URL permanently
setx DB_URL "%DB_URL%" >nul

echo ✅ Loaded DB configuration:
echo    DB_URL=%DB_URL%
echo    DB_USER=%DB_USER%

cd /d "D:\sms spam\Data_Base"

REM ✅ Compile SpamDatabase.java
javac -cp ".;postgresql-42.7.3.jar;json.jar" SpamDatabase.java

REM ✅ TEST DB connection interactively (optional)
java -cp ".;postgresql-42.7.3.jar;json.jar" ^
    -DDB_URL="%DB_URL%" -DDB_USER="%DB_USER%" -DDB_PASSWORD="%DB_PASSWORD%" ^
    SpamDatabase

REM === Now start Java Backend with SAME DB credentials ===

REM === Start Java Backend in background ===
start /b cmd /c "cd /d D:\sms spam\Backend && javac -cp .;..\Data_Base;..\Data_Base\postgresql-42.7.3.jar;..\Data_Base\json.jar SpamCollectorServer.java && java -cp .;..\Data_Base;..\Data_Base\postgresql-42.7.3.jar;..\Data_Base\json.jar SpamCollectorServer"
echo ✅ Java Backend started...

pause
