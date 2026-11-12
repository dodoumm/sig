
@echo off
chcp 65001 > nul
setlocal EnableExtensions
title Git Commit and Push

:: BAT 파일이 있는 폴더로 이동
cd /d "%~dp0"

echo ======================================
echo          GIT COMMIT AND PUSH
echo ======================================
echo.

:: Git 저장소 확인
git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git 저장소가 아닙니다.
    goto ERROR
)

:: 커밋 메시지 입력
:INPUT
set "MESSAGE="
set /p "MESSAGE=커밋 내용을 입력하세요: "

if not defined MESSAGE (
    echo [ERROR] 커밋 내용이 비어 있습니다.
    goto INPUT
)

echo.
echo 커밋 메시지: %MESSAGE%
echo.

:: 실행 확인
choice /C YN /M "커밋 및 Push를 진행할까요?"
if errorlevel 2 goto CANCEL

:: Git Add
echo.
echo [1/3] Git Add...
git add .
if errorlevel 1 goto ERROR

:: Git Commit
echo.
echo [2/3] Git Commit...
git commit -m "%MESSAGE%"
if errorlevel 1 goto ERROR

:: Git Push
echo.
echo [3/3] Git Push...
git push origin main
if errorlevel 1 goto ERROR

echo.
echo ======================================
echo          PUSH SUCCESS!
echo ======================================
pause
exit /b 0

:CANCEL
echo 작업을 취소했습니다.
pause
exit /b 0

:ERROR
echo.
echo ======================================
echo             GIT ERROR
echo ======================================
git status
pause
exit /b 1
