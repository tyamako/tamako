@echo off
rem 顔隠しの確認と修正。危険な箇所を抜き出して見せ、その場で直す。
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
echo  顔隠しの確認と修正
echo ============================================
echo.
echo 要確認の箇所だけを抜き出して書き出します（素材と同じ画質・短い動画）。
echo.

python -m tamako remask --config config.json
if errorlevel 1 (
    echo.
    echo 途中で止まりました。上のメッセージを確認してください。
    pause
    exit /b 1
)

echo.
echo output\_review\review.mp4 を再生して、隠しきれていない箇所が無いか見てください。
echo （このフォルダには素顔が写っています。人に渡さないでください）
echo.
set "ANSWER="
set /p ANSWER="直すところがありますか？ (y = 修正の窓を開く / それ以外 = 終了): "
if /i not "%ANSWER%"=="y" (
    echo.
    echo 終了しました。声を録ったら scripts\2_finish.bat を実行してください。
    pause
    exit /b 0
)

echo.
echo 窓を開きます。操作方法は画面の下に出ます。
echo   左ドラッグ = 箱を足す / 右クリック = 箱を消す / + - = 大きさ
echo   c = 確認済み / u = 元に戻す / n p = 次と前 / q = 終了
echo.
python -m tamako fix --config config.json
if errorlevel 1 (
    echo.
    echo 窓を開けませんでした。上のメッセージを確認してください。
    pause
    exit /b 1
)

echo.
echo ============================================
echo  修正を反映します
echo ============================================
echo.
python -m tamako edit --config config.json
if errorlevel 1 (
    echo.
    echo 書き出しに失敗しました。上のメッセージを確認してください。
    pause
    exit /b 1
)

echo.
echo 直したものが output\edited.mp4 に反映されました。
echo まだ気になる箇所があれば、この 1b_fix.bat をもう一度実行してください。
echo.
pause
