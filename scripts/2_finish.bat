@echo off
rem 後半の工程: 収録した声を文字起こしして、字幕を焼き込み音声を差し替える。
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
echo  2. 文字起こし / 字幕 / 音声の差し替え
echo ============================================
echo.

set "VIDEO=output\edited.mp4"
if not exist "%VIDEO%" (
    echo [エラー] %VIDEO% がありません。先に scripts\1_edit.bat を実行してください。
    pause
    exit /b 1
)

echo 収録した音声ファイルを、この窓にドラッグ＆ドロップして Enter を押してください。
echo （wav / mp3 / m4a など）
echo.
set "AUDIO="
set /p AUDIO="音声ファイル: "

rem ドラッグ＆ドロップで付く引用符を外す。
set AUDIO=%AUDIO:"=%

if "%AUDIO%"=="" (
    echo.
    echo 音声ファイルが指定されていません。
    pause
    exit /b 1
)
if not exist "%AUDIO%" (
    echo.
    echo [エラー] ファイルが見つかりません: %AUDIO%
    pause
    exit /b 1
)

echo.
echo 文字起こしをしています。初回はモデルの取得に時間がかかります...
echo.
python -m tamako finish --config config.json --video "%VIDEO%" --audio "%AUDIO%"
if errorlevel 1 (
    echo.
    echo 失敗しました。上のメッセージを確認してください。
    pause
    exit /b 1
)

echo.
echo ============================================
echo  完成しました: output\final.mp4
echo ============================================
echo.
echo 字幕の文字を直したい場合は output\subtitles.srt をメモ帳で編集してから、
echo scripts\3_resubtitle.bat を実行すると焼き直せます。
echo.
pause
