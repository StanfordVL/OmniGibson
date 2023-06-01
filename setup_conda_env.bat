@echo off

:: Make sure that the ISAAC_SIM_PATH variable is set
if  "%ISAAC_SIM_PATH%" == "" (
    echo "Please set ISAAC_SIM_PATH!" 
    exit /b
)

:: Sanity check whether env variable is set correctly 
if not exist %ISAAC_SIM_PATH%\isaac-sim.bat (
    echo isaac-sim.bat not found in %ISAAC_SIM_PATH%! Make sure you have set the correct ISAAC_SIM_PATH
    exit /b
)

:: Choose venv name
set conda_name=omnigibson
echo The new conda environment will be named [4momnigibson[0m by default.
set /p conda_name=If you want to use a different name, please type in here (press enter to skip): 
echo Using %conda_name% as the conda environment name

:: Create a conda environment with python 3.7
call conda create -y -n %conda_name% python=3.7 || goto :error
call conda activate %conda_name% || goto :error

:: We add some preprocessing information so that the Isaac Sim paths are linked to this environment upon startup
:: See https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux for reference
mkdir %CONDA_PREFIX%\etc\conda\activate.d
mkdir %CONDA_PREFIX%\etc\conda\deactivate.d
type NUL>%CONDA_PREFIX%\etc\conda\activate.d\env_vars.bat
type NUL>%CONDA_PREFIX%\etc\conda\deactivate.d\env_vars.bat
type NUL>%CONDA_PREFIX%\etc\conda\activate.d\env_vars.ps1
type NUL>%CONDA_PREFIX%\etc\conda\deactivate.d\env_vars.ps1

:: Add support for cmd
set CONDA_ACT_FILE_CMD=%CONDA_PREFIX%\etc\conda\activate.d\env_vars.bat
echo @echo off>>%CONDA_ACT_FILE_CMD%
echo set PYTHONPATH_OLD=%%PYTHONPATH%%>>%CONDA_ACT_FILE_CMD%
echo set PYTHONPATH=%%PYTHONPATH%%;%ISAAC_SIM_PATH%\site>>%CONDA_ACT_FILE_CMD%
echo set CARB_APP_PATH=%ISAAC_SIM_PATH%\kit>>%CONDA_ACT_FILE_CMD%
echo set EXP_PATH=%ISAAC_SIM_PATH%\apps>>%CONDA_ACT_FILE_CMD%
echo set ISAAC_PATH=%ISAAC_SIM_PATH%>>%CONDA_ACT_FILE_CMD%

set CONDA_DEACT_FILE_CMD=%CONDA_PREFIX%\etc\conda\deactivate.d\env_vars.bat
echo @echo off>>%CONDA_DEACT_FILE_CMD%
echo set PYTHONPATH=%%PYTHONPATH_OLD%%>>%CONDA_DEACT_FILE_CMD%
echo set PYTHONPATH_OLD="">>%CONDA_DEACT_FILE_CMD%

:: Add support for powershell
set CONDA_ACT_FILE_PWSH=%CONDA_PREFIX%\etc\conda\activate.d\env_vars.ps1
echo $env:PYTHONPATH_OLD="$env:PYTHONPATH">>%CONDA_ACT_FILE_PWSH%
echo $env:PYTHONPATH="$env:PYTHONPATH;%ISAAC_SIM_PATH%\site">>%CONDA_ACT_FILE_PWSH%
echo $env:CARB_APP_PATH="%ISAAC_SIM_PATH%\kit">>%CONDA_ACT_FILE_PWSH%
echo $env:EXP_PATH="%ISAAC_SIM_PATH%\apps">>%CONDA_ACT_FILE_PWSH%
echo $env:ISAAC_PATH="%ISAAC_SIM_PATH%">>%CONDA_ACT_FILE_PWSH%

set CONDA_DEACT_FILE_PWSH=%CONDA_PREFIX%\etc\conda\deactivate.d\env_vars.ps1
echo $env:PYTHONPATH="$env:PYTHONPATH_OLD">>%CONDA_DEACT_FILE_PWSH%
echo $env:PYTHONPATH_OLD="$null">>%CONDA_DEACT_FILE_PWSH%


:: Install omnigibson!
call pip install -e . || goto :error

:: Cycle conda environment so that all dependencies are propagated
call conda deactivate || goto :error
goto :end 

:error
echo:
echo An error occurred during installation. Please check the error message above.
echo:
exit /b

:end
echo:
echo OmniGibson successfully installed! Please run [4mconda activate %conda_name%[0m to activate the environment.
echo:
