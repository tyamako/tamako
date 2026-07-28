@echo off
rem 字幕(SRT)を手で直したあと、焼き込みだけをやり直す。
rem 文字起こしは走らないので速い。
chcp 65001 > nul
setlocal
cd /d "%~dp0.."

if not exist ".venv\Scripts\activate.bat" (
    echo [エラー] 先に scripts\setup.bat を実行してください。
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"

echo ============================================
echo  字幕を直して焼き直す
echo ============================================
echo.

set "VIDEO=output\edited.mp4"
set "SRT=output\subtitles.srt"

if not exist "%SRT%" (
    echo [エラー] %SRT% がありません。先に scripts\2_finish.bat を実行してください。
    pause
    exit /b 1
)

echo 使う字幕: %SRT%
echo.
echo 収録した音声ファイルを、この窓にドラッグ＆ドロップして Enter を押してください。
echo （音声を差し替えない場合は、何も入れずに Enter）
echo.
set "AUDIO="
set /p AUDIO="音声ファイル: "
set AUDIO=%AUDIO:"=%

echo.
echo 焼き込んでいます...
echo.
if "%AUDIO%"=="" (
    python -m tamako subtitle --config config.json --video "%VIDEO%" --srt "%SRT%"
) else (
    python -m tamako subtitle --config config.json --video "%VIDEO%" --srt "%SRT%" --audio "%AUDIO%"
)
if errorlevel 1 (
    echo.
    echo 失敗しました。上のメッセージを確認してください。
    pause
    exit /b 1
)

echo.
echo 完成しました: output\final.mp4
echo.
pause
