@echo off
rem 前半の工程: 撮影順に並べ、不要な区間を切り、顔を PNG で隠す。
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
echo  1. 並べる / 切る / 顔を隠す
echo ============================================
echo.
echo まず、どう切られるかを確認します（まだ書き出しません）。
echo.

python -m tamako check --config config.json
if errorlevel 1 (
    echo.
    echo 途中で止まりました。上のメッセージを確認してください。
    pause
    exit /b 1
)

echo.
set "ANSWER="
set /p ANSWER="この内容で書き出しますか？ (y = 実行 / それ以外 = 中止): "
if /i not "%ANSWER%"=="y" (
    echo.
    echo 中止しました。config.json の cut を調整してから、もう一度実行してください。
    pause
    exit /b 0
)

echo.
echo 書き出しています。素材の長さによっては時間がかかります...
echo.
python -m tamako edit --config config.json
if errorlevel 1 (
    echo.
    echo 書き出しに失敗しました。上のメッセージを確認してください。
    pause
    exit /b 1
)

echo.
echo ============================================
echo  つぎにやること
echo ============================================
echo.
echo   1. 上に「要確認のサイト」が出ていたら scripts\1b_fix.bat を実行する
echo      （危険な箇所だけを抜き出して見せ、その場で直せます）
echo   2. output\edited.mp4 を再生して確認する
echo      ※ 顔が隠しきれていない箇所が無いか、必ず全編を目で確かめてください
echo   3. その映像を見ながら声を録音する
echo   4. 録音した音声ファイルを用意して scripts\2_finish.bat を実行する
echo.
pause
