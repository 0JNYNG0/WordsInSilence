@echo off
cd /d "%~dp0"
"research\front_keyphase_v1\ops\word100_signrep_feasibility_20261029\runtime\Scripts\python.exe" -X utf8 -B "frontal_demo_launcher.py"
if errorlevel 1 pause
