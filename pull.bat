
@echo off
chcp 65001 > nul
title WebProject Main Pull

echo       WEBPROJECT GIT PULL
echo.

:: Git 저장소 확인
git rev-parse --is-inside-work-tree > nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git 저장소가 아닙니다.
    pause
    exit /b 1
)

:: Remote 확인
echo [1/2] Remote 확인...
git remote get-url origin
if errorlevel 1 (
    echo [ERROR] Origin이 설정되지 않았습니다.
    pause
    exit /b 1
)

:: Pull 실행
echo.
echo [2/2] Main Branch Pull...
git pull origin main

if errorlevel 1 (
    echo.
    echo [ERROR] Pull 실패!
    pause
    exit /b 1
)

echo.
echo          PULL SUCCESS!
pause
