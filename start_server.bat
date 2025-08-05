@echo on
REM Navigate to the directory where this script is located
cd /d "%~dp0"

echo.
echo Activating virtual environment...
CALL .\django_venv\Scripts\activate

echo.
echo Starting Django development server in a NEW window...
echo You will need to keep that new window open for the server to run.
echo Press Ctrl+C in that NEW window to stop the server.

REM Start the Django server in a new command prompt window
REM "Django Server" is the title of the new window
REM /k keeps the new command prompt window open after running the command
start "Django Server" cmd /k "python manage.py runserver"

echo.
echo Waiting a few seconds for the server to initialize...
REM Add a 5-second delay. /nobreak prevents stopping the timeout with a key press. >nul suppresses output.
timeout /t 5 /nobreak >nul

echo Opening the application in your default browser...
start http://127.0.0.1:8000/sentiment/

echo.
echo This window will close automatically after a short delay.
echo Please keep the "Django Server" window open.
timeout /t 3 >nul

REM The PAUSE at the very end is optional now. If you want this main window to close automatically, remove PAUSE.
REM If you want it to wait for you to see the "Opening..." message, keep it.
PAUSE