@echo off
title TradeSage Auto-Trainer
echo ===================================================
echo   TradeSage Daily Auto-Training Scheduler
echo ===================================================
echo.
echo Starting the background scheduler...
echo It will automatically train the model daily at 18:00
echo Do not close this window to keep the scheduler running.
echo.
python auto_train_daily.py
pause
