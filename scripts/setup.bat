@echo off
rem 初回だけ実行する。Python の仮想環境を作って必要なものを入れる。
chcp 65001 > nul
setlocal
cd /d "%~dp0.."

echo ============================================
echo  tamako セットアップ
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [エラー] Python が見つかりません。
    echo.
    echo   https://www.python.org/downloads/windows/ から Python 3.10 以降を入れて、
    echo   インストール時に "Add python.exe to PATH" に必ずチェックを入れてください。
    echo.
    pause
    exit /b 1
)

echo Python を確認しました:
python --version
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo 仮想環境を作っています...
    python -m venv .venv
    if errorlevel 1 (
        echo [エラー] 仮想環境を作れませんでした。
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"

echo 必要なものを入れています。数分かかります...
echo.
python -m pip install --upgrade pip --quiet
python -m pip install -e ".[subtitle]"
if errorlevel 1 (
    echo.
    echo [エラー] インストールに失敗しました。上のメッセージを確認してください。
    pause
    exit /b 1
)

echo.
echo 顔検出モデルを取得しています...
python -c "from tamako.faces import ensure_model; print('  ->', ensure_model())"
if errorlevel 1 (
    echo.
    echo [注意] 顔検出モデルを取得できませんでした。
    echo        ネットワークを確認して、もう一度この setup.bat を実行してください。
    pause
    exit /b 1
)

if not exist "config.json" (
    echo.
    echo 設定ファイルを作っています...
    python -m tamako init .
)

echo.
echo ============================================
echo  完了しました
echo ============================================
echo.
echo つぎにやること:
echo   1. input フォルダに撮影した動画を入れる
echo   2. 顔に重ねる PNG を mask.png という名前でこのフォルダに置く
echo   3. scripts\1_edit.bat をダブルクリックする
echo.
pause
